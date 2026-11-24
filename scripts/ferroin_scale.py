"""Iron-scale correction for every ferroin standard and spike.

jdc, 2026-10-02: nominal concentrations give ferroin eps(510) = 19,678, 1.77x
the literature. The excess is a single scale on the stock, so concentrations are
calculated as for FeSO4.H2O: 278.01 / 169.92 = 1.636 times the label. The salt
looks pale blue, possibly whitish (Q7.1), so water loss is a possible cause, not
an established one; the basis is agreement of eps with the literature.
Evidence: Data/analysis/wine_ferroin/epsilon_diagnostic/findings.md. The
corrected eps(510) is ~12,030 M-1 cm-1, between the classic 11,100 and
Smith et al. 2021's 12,450. If Q7.3 gives a measured ratio, use it here.
"""

M_HEPTAHYDRATE = 151.90 + 7 * 18.015
M_MONOHYDRATE = 151.90 + 1 * 18.015
FE_SCALE = M_HEPTAHYDRATE / M_MONOHYDRATE


def true_uM(labelled_uM):
    return labelled_uM * FE_SCALE
