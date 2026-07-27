#include "ife_fh_core.h"

using namespace mcl::bn;

iFE_FH_Core::iFE_FH_Core(int m_rank, const G1& g1, const G2& g2)
    : m_dim(m_rank), g1_base(g1), g2_base(g2) {
    N = 2 * m_dim + 2;
    s1.resize(N);
    s2.resize(N + 1);
}

void iFE_FH_Core::u_setup() {
    for (int i = 0; i < N; ++i) s1[i].setByCSPRNG();
    for (int i = 0; i <= N; ++i) s2[i].setByCSPRNG();
}

// KeyGen precompute (B side). Everything that does not depend on the B-row
// factors: r2, k1 = g2^{-r2}, the t_lb^{s1[2R]} term, and all structural k2
// bases. Online folds in the R real factors (slots 0..R-1).
iFE_FH_Core::KeyPrecomp iFE_FH_Core::keygen_precompute(const G2& t_lb) {
    Fr r2; r2.setByCSPRNG();
    return keygen_precompute(t_lb, r2);
}

iFE_FH_Core::KeyPrecomp iFE_FH_Core::keygen_precompute(const G2& t_lb, const Fr& r2) {
    KeyPrecomp pre;
    Fr neg_r2; Fr::neg(neg_r2, r2); G2::mul(pre.k1, g2_base, neg_r2);

    // k2[0] base = g2^{r2 s2[0]} + t_lb^{s1[2R]}  (online adds g2^{sum s1[k] b_k}).
    Fr k2_0_scalar = r2 * s2[0];
    G2 g_part, t_part;
    G2::mul(g_part, g2_base, k2_0_scalar);
    G2::mul(t_part, t_lb, s1[2 * m_dim]);
    G2::add(pre.k2_0_base, g_part, t_part);

    pre.k2_base.resize(N);
    for (int i = 0; i < N; ++i) {
        Fr exp = r2 * s2[i + 1];
        G2 k2_i; G2::mul(k2_i, g2_base, exp);
        if (i == 2 * m_dim) G2::add(k2_i, k2_i, t_lb);  // reserved t_lb slot
        pre.k2_base[i] = k2_i;                          // real slots get g2^{b_k} online
    }
    return pre;
}

// KeyGen online. b_real: the R real B-row factors (slots 0..R-1).
std::pair<G2, std::vector<G2>> iFE_FH_Core::keygen_online(const KeyPrecomp& pre, const std::vector<Fr>& b_real) {
    std::vector<G2> k2; k2.reserve(N + 1);

    Fr s1_dot_b = 0;
    for (int k = 0; k < m_dim; ++k) s1_dot_b += s1[k] * b_real[k];
    G2 g_part, k2_0; G2::mul(g_part, g2_base, s1_dot_b); G2::add(k2_0, pre.k2_0_base, g_part);
    k2.push_back(k2_0);

    for (int i = 0; i < N; ++i) {
        G2 k2_i = pre.k2_base[i];
        if (i < m_dim) { G2 gb; G2::mul(gb, g2_base, b_real[i]); G2::add(k2_i, k2_i, gb); }
        k2.push_back(k2_i);
    }
    return std::make_pair(pre.k1, k2);
}

// Dual-mode KeyGen for a zero row: compact single element k_hat = t_lb^{s1[2R]} in G2.
G2 iFE_FH_Core::keygen_zero(const G2& t_lb) {
    G2 k_hat;
    G2::mul(k_hat, t_lb, s1[2 * m_dim]);
    return k_hat;
}

// Enc precompute (A side). a_offline (length N) carries only the data-independent
// entries (the x mask at slot 2R); r1 is sampled and baked into every base term.
// Online folds g1^{a_k} into slots 1..R and corrects c1.
iFE_FH_Core::EncPrecomp iFE_FH_Core::encrypt_precompute(const std::vector<Fr>& a_offline) {
    Fr r1; r1.setByCSPRNG();
    return encrypt_precompute(a_offline, r1);
}

iFE_FH_Core::EncPrecomp iFE_FH_Core::encrypt_precompute(const std::vector<Fr>& a_offline, const Fr& r1) {
    EncPrecomp pre;
    std::vector<Fr> c2_zr(N + 1, 0);
    Fr::neg(c2_zr[0], r1);
    for (int i = 0; i < N; ++i) c2_zr[i + 1] = r1 * s1[i] + a_offline[i];

    pre.c2_base.resize(N + 1);
    for (int i = 0; i <= N; ++i) G1::mul(pre.c2_base[i], g1_base, c2_zr[i]);

    Fr s2_dot = 0;
    for (int i = 0; i <= N; ++i) s2_dot += s2[i] * c2_zr[i];
    G1::mul(pre.c1_base, g1_base, s2_dot);
    return pre;
}

// Enc online. a_real: the R real A-column factors (slots 0..R-1).
std::pair<G1, std::vector<G1>> iFE_FH_Core::encrypt_online(const EncPrecomp& pre, const std::vector<Fr>& a_real) {
    std::vector<G1> c2 = pre.c2_base;
    Fr delta = 0;
    for (int k = 0; k < m_dim; ++k) {
        G1 ga; G1::mul(ga, g1_base, a_real[k]); G1::add(c2[k + 1], c2[k + 1], ga);
        delta += s2[k + 1] * a_real[k];
    }
    G1 c1, gd; G1::mul(gd, g1_base, delta); G1::add(c1, pre.c1_base, gd);
    return std::make_pair(c1, c2);
}
