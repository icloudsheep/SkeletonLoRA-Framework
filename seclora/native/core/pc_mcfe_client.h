#ifndef PC_MCFE_CLIENT_H
#define PC_MCFE_CLIENT_H

#include <vector>
#include <string>
#include <utility>
#include <mcl/bn.hpp>
#include "seclora_types.h"
#include "ife_fh_core.h"

// Data-independent precompute for one encA call (per column).
struct EncA_Precomp {
    std::vector<iFE_FH_Core::EncPrecomp> ife;  // iFE ciphertext base (r1, x baked in)
    std::vector<mcl::bn::G1> c_i;              // u_s_term + g0^x  (non-zero column mask)
    std::vector<mcl::bn::G1> u_s_only;         // u_s_term         (zero column mask)
    std::vector<mcl::bn::Fr> x;                // per-column offline randomness
};

// Data-independent precompute for one encB call (per row).
struct EncB_Precomp {
    std::vector<iFE_FH_Core::KeyPrecomp> key;  // iFE key base (r2, t_lb baked in)
    std::vector<mcl::bn::G2> k_hat;            // compact dual-mode key for a zero row
};

// PC-MCFE client: holds a long-term key s_i, encrypts the A matrix (encA),
// generates B-row keys (encB), and contributes its KeyGen share.
class PC_MCFE_Client {
private:
    int client_id;
    int m_rank;
    const SecLoRA_PP& pp;

    std::vector<mcl::bn::Fr> s_i;   // long-term key s_i in Z_p^2
    mcl::bn::Fr r_si;               // blinding for C_{s,i}
    mcl::bn::G1 C_si;               // C_{s,i} = g0*s0 + g1*s1 + h*r_si

    std::vector<mcl::bn::Fr> last_x_randomness;   // per-column A-side randomness x_v

    std::vector<std::vector<mcl::bn::Fr>> local_A;  // R x cols
    std::vector<std::vector<mcl::bn::Fr>> local_B;  // rows x R

    iFE_FH_Core ife;

    EncA_Precomp encA_pre;   // filled by precompute_encA, consumed by encA
    EncB_Precomp encB_pre;   // filled by precompute_encB, consumed by encB

public:
    PC_MCFE_Client(int id, int rank, const SecLoRA_PP& global_pp);
    mcl::bn::G1 GetCommitment() const { return C_si; }

    std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr> KeyGenShare(const std::vector<mcl::bn::Fr>& p_weights);

    void u_setup() { ife.u_setup(); }

    // Inject real quantized integer factors. A: rank x cols, B: rows x rank.
    void SetLoraMatrices(const std::vector<std::vector<long long>>& A,
                         const std::vector<std::vector<long long>>& B);
    void SetLoraMatricesFr(
        const std::vector<std::vector<mcl::bn::Fr>>& A,
        const std::vector<std::vector<mcl::bn::Fr>>& B);

    // Selective encryption: build this client's PLAINTEXT share for server S_P.
    // Only the plaintext region is included -- encrypted B rows [0,eb) and encrypted
    // A cols [0,ea) are zeroed out, so S_P never receives the encrypted factors.
    // Bp: m x R (rows < eb zeroed); Ap: R x n (cols < ea zeroed).
    void BuildPlaintextShare(int eb, int ea,
                             std::vector<std::vector<mcl::bn::Fr>>& Bp,
                             std::vector<std::vector<mcl::bn::Fr>>& Ap) const;

    // Multithreaded encryption: randomness is sampled serially, then the
    // group-op-heavy precompute/online work runs across `threads`.
    void precompute_encA_indices_mt(int layer_id, int pos_y, int round_q, int matrix_cols,
                                    const std::vector<int>& cols, int threads);
    void precompute_encB_indices_mt(int layer_id, int pos_y, int round_q, int matrix_rows,
                                    const std::vector<int>& rows, int threads);
    std::vector<A_Ciphertext_Slot> encA_indices_mt(int layer_id, int pos_y, int round_q, int matrix_cols,
                                                   const std::vector<int>& cols, int threads);
    std::vector<B_SecretKey_Slot> encB_indices_mt(int layer_id, int pos_y, int round_q, int matrix_rows,
                                                  const std::vector<int>& rows, int threads);

    // Online encryption no longer needs the group elements generated offline.
    // Release them before moving to the next client to cap FULL+SK peak memory.
    void ClearEncryptionPrecompute();

    // Expected plaintext cell value (B*A)[u][v], for correctness checking.
    mcl::bn::Fr GetCellExpectedProduct(int u, int v) const;
};

#endif  // PC_MCFE_CLIENT_H
