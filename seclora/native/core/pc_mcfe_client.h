#ifndef PC_MCFE_CLIENT_H
#define PC_MCFE_CLIENT_H

#include <array>
#include <vector>
#include <string>
#include <utility>
#include <mcl/bn.hpp>
#include "seclora_types.h"
#include "ife_fh_core.h"

// Data-independent precompute for one encA call (per column).
struct EncA_Precomp {
    std::vector<iFE_FH_Core::EncPrecomp> ife;
    std::vector<DmcfeCiphertext2> dfe;
    std::vector<std::array<mcl::bn::Fr, 2>> x;
};

// Data-independent precompute for one encB call (per row).
struct EncB_Precomp {
    std::vector<iFE_FH_Core::KeyPrecomp> key;
};

// PC-DMCFE client: FH-IPFE is round-specific; DMCFE state is long-lived.
class PC_MCFE_Client {
private:
    int client_id;
    int m_rank;
    const SecLoRA_PP& pp;
    const DmcfePublicParams2& dfe_pp;
    DmcfeClientSecret2 dfe_secret;

    std::vector<std::vector<mcl::bn::Fr>> local_A;  // R x cols
    std::vector<std::vector<mcl::bn::Fr>> local_B;  // rows x R

    iFE_FH_Core ife;

    EncA_Precomp encA_pre;   // filled by precompute_encA, consumed by encA
    EncB_Precomp encB_pre;   // filled by precompute_encB, consumed by encB

public:
    PC_MCFE_Client(int id, int rank, const SecLoRA_PP& global_pp,
                   const DmcfePublicParams2& global_dfe_pp,
                   const DmcfeClientSecret2& client_dfe_secret);

    DmcfeKeyShare2 KeyGenShare(
        const std::vector<mcl::bn::Fr>& p_weights) const;

    void u_setup() { ife.u_setup(); }

    // Inject real quantized integer factors. A: rank x cols, B: rows x rank.
    void SetLoraMatrices(const std::vector<std::vector<long long>>& A,
                         const std::vector<std::vector<long long>>& B);

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
