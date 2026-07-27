"""CKKS 密钥上下文工具。

本文件复制自 /Users/bilibili/Github/SkeletonLoRA/fe_context.py，仅调整包内导入方式。
"""

import tenseal as ts


def create_secret_context(poly_modulus_degree, coeff_mod_bit_sizes, global_scale,
                          galois=False):
    """创建含私钥的 CKKS context。"""
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=poly_modulus_degree,
                     coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    ctx.global_scale = global_scale
    if galois:
        ctx.generate_galois_keys()
    return ctx


def derive_public_context(secret_ctx):
    """从私钥 context 派生去掉私钥的公开 context。"""
    pub = secret_ctx.copy()
    pub.make_context_public()
    return pub
