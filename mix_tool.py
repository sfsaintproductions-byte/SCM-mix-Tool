import streamlit as st
import numpy as np
import pandas as pd
import os

# ==========================================
# 1. SOLVER LOGIC
# ==========================================

def get_dry_masses(target_Al_Si, target_Ca_Si,
                   nCa_cem, nSi_fa, nAl_fa, nSi_sf, nAl_sf,
                   use_fa, use_sf):
    """
    Helper function to calculate ONLY the solid masses (Cem, FA, SF)
    This allows us to back-calculate the required water for a specific W/B.
    """
    if use_fa and use_sf:
        F = np.array([
            [0, nAl_fa, nAl_sf],
            [0, nSi_fa, nSi_sf],
            [nCa_cem, 0, 0]
        ])
        R = np.array([target_Al_Si, 1.0, target_Ca_Si])
        try:
            return np.linalg.solve(F, R)
        except np.linalg.LinAlgError:
            return None
    elif use_fa and not use_sf:
        if nSi_fa == 0: return None
        w_fa = 1.0 / nSi_fa
        w_sf = 0.0
        w_cem = target_Ca_Si / nCa_cem
        return np.array([w_cem, w_fa, w_sf])
    elif use_sf and not use_fa:
        if nSi_sf == 0: return None
        w_sf = 1.0 / nSi_sf
        w_fa = 0.0
        w_cem = target_Ca_Si / nCa_cem
        return np.array([w_cem, w_fa, w_sf])
    return None


def solve_mix(target_Al_Si, target_Ca_Si, target_H_Si,
              nCa_cem,
              nSi_fa, nAl_fa,
              nSi_sf, nAl_sf,
              nH_water,
              use_fa, use_sf):
    if use_fa and use_sf:
        F = np.array([
            [0, nAl_fa, nAl_sf, 0],
            [0, nSi_fa, nSi_sf, 0],
            [nCa_cem, 0, 0, 0],
            [0, 0, 0, nH_water]
        ])
        R = np.array([target_Al_Si, 1.0, target_Ca_Si, target_H_Si])
        try:
            return np.linalg.solve(F, R), "Success"
        except:
            return None, "Matrix Singular"
    elif use_fa and not use_sf:
        if nSi_fa == 0: return None, "No Silica"
        w_fa = 1.0 / nSi_fa
        w_sf = 0.0
        w_cem = target_Ca_Si / nCa_cem
        w_water = target_H_Si / nH_water
        return np.array([w_cem, w_fa, w_sf, w_water]), "Success"
    elif use_sf and not use_fa:
        if nSi_sf == 0: return None, "No Silica"
        w_sf = 1.0 / nSi_sf
        w_fa = 0.0
        w_cem = target_Ca_Si / nCa_cem
        w_water = target_H_Si / nH_water
        return np.array([w_cem, w_fa, w_sf, w_water]), "Success"
    return None, "Select SCM"


# ==========================================
# 2. APP INTERFACE
# ==========================================

st.set_page_config(page_title="SCM Mix Designer", layout="wide")

st.title("🧪 SCM Mix Tool")
st.markdown("Hendrik sniffs glue")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Feedstock Selection")
    use_fa = st.checkbox("Include Fly Ash", value=True)
    use_sf = st.checkbox("Include Silica Fume", value=True)

    st.markdown("---")
    st.header("2. Chemistry")

    with st.expander("Binder Composition", expanded=True):
        st.subheader("GP Cement")
        c3s = st.number_input("C3S (%)", value=11.8, step=0.1, min_value=0.0)
        c2s = st.number_input("C2S (%)", value=41.0, step=0.1, min_value=0.0)
        
        # --- NEW SECTION: CH YIELDS ---
        with st.expander("⚙️ Modify CH Yield Constants", expanded=False):
            yield_c3s = st.number_input("Yield C3S (g CH/g)", value=0.252, step=0.001, format="%.3f")
            yield_c2s = st.number_input("Yield C2S (g CH/g)", value=0.039, step=0.001, format="%.3f")
            
        nCa_cem = (c3s * yield_c3s + c2s * yield_c2s) / 74.093
        st.caption(f"Yield: {nCa_cem:.4f} mol CH/100g")

        nSi_fa, nAl_fa, ratio_fa = 0, 0, 0
        if use_fa:
            st.markdown("---")
            st.subheader("Fly Ash")
            fa_si_wt = st.number_input("FA Si (wt%)", value=13.40, min_value=0.0)
            fa_al_wt = st.number_input("FA Al (wt%)", value=6.12, min_value=0.0)
            nSi_fa = fa_si_wt / 28.085
            nAl_fa = fa_al_wt / 26.982
            ratio_fa = nAl_fa / nSi_fa if nSi_fa > 0 else 0

        nSi_sf, nAl_sf, ratio_sf = 0, 0, 0
        if use_sf:
            st.markdown("---")
            st.subheader("Silica Fume")
            sf_si_wt = st.number_input("SF Si (wt%)", value=46.74, min_value=0.0)
            sf_al_wt = st.number_input("SF Al (wt%)", value=0.0, min_value=0.0)
            nSi_sf = sf_si_wt / 28.085
            nAl_sf = sf_al_wt / 26.982
            ratio_sf = nAl_sf / nSi_sf if nSi_sf > 0 else 0

    with st.expander("Filler Properties", expanded=True):
        st.subheader("Sand")
        sand_abs = st.number_input("Absorption Capacity (%)", value=0.39, step=0.01, min_value=0.0)

    st.markdown("---")

# --- DETERMINE LOGIC & BOUNDS ---

if use_fa and use_sf:
    min_al_si = min(ratio_fa, ratio_sf)
    max_al_si = max(ratio_fa, ratio_sf)
    default_al_si = 0.30
    if default_al_si < min_al_si: default_al_si = min_al_si
    if default_al_si > max_al_si: default_al_si = max_al_si
    current_fixed_ratio = None
elif use_fa:
    current_fixed_ratio = ratio_fa
elif use_sf:
    current_fixed_ratio = ratio_sf
else:
    current_fixed_ratio = 0

# --- MAIN DASHBOARD ---

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("2. Target Ratios")

    # AL/SI
    if use_fa and use_sf:

        t_al_si = st.number_input("Al / Si (molar)",
                                  min_value=float(min_al_si), max_value=float(max_al_si), value=float(default_al_si),
                                  step=0.005, format="%.4f")
    elif use_fa or use_sf:
        st.warning(f"🔒 Ratio fixed by SCM chemistry")
        t_al_si = st.number_input("Al / Si (molar)", value=float(current_fixed_ratio), disabled=True, format="%.4f")
    else:
        t_al_si = 0
        st.error("Select an SCM.")

    # CA/SI
    t_ca_si = st.number_input("Ca / Si (molar)", min_value=0.0000, value=0.0318, step=0.0010, format="%.4f")

    # WATER CONTROL
    st.write("")
    water_mode = st.radio("Water Control Method:", ["W/B Ratio", "H/Si Ratio"], horizontal=True)
    nH_water_const = 11.102

    if water_mode == "W/B Ratio":
        # Show W/B Input
        target_wb = st.number_input("Target W/B", min_value=0.1, value=0.40, step=0.01)

        # Back-calculate H/Si
        dry_weights = get_dry_masses(t_al_si, t_ca_si, nCa_cem, nSi_fa, nAl_fa, nSi_sf, nAl_sf, use_fa, use_sf)
        if dry_weights is not None:
            total_solids_parts = np.sum(dry_weights)
            req_water_parts = total_solids_parts * target_wb
            calculated_h_si = req_water_parts * nH_water_const
        else:
            calculated_h_si = 0.0

        # Show H/Si as DISABLED (Read-Only)
        t_h_si = st.number_input("H / Si (molar) [Calculated]", value=float(calculated_h_si), disabled=True,
                                 format="%.2f")

    else:
        # Show H/Si as ACTIVE Input
        t_h_si = st.number_input("H / Si (molar)", min_value=0.00, value=5.72, step=0.10, format="%.2f")

with col2:
    st.subheader("3. Batch Sizing")
    batch_binder_kg = st.number_input("Total Binder Mass (kg)", value=5.0, step=0.5, min_value=0.1)
    sand_binder_ratio = st.number_input("Sand : Binder Ratio", value=3.0, step=0.1, min_value=0.0)

with col3:
    st.subheader("4. Lab Conditions")
    sand_moisture = st.number_input("Sand Moisture Content (%)", value=2.0, step=0.1, min_value=0.0)

    st.markdown("---")
    st.write("**Admixtures (mL / 100kg Binder)**")

    lock_wb = st.checkbox("Lock Target W/B (Compensate for Admix)", value=True,
                          help="If Checked: Batch water is reduced so Total Water (Batch + Admix) stays equal to Target W/B.")
    adm_wr = st.number_input("Water Reducer", value=0, step=50, min_value=0)
    adm_vm = st.number_input("Viscosity Mod.", value=0, step=50, min_value=0)
    adm_accel = st.number_input("Accelerator", value=0, step=50, min_value=0)

    # --- MICROFIBRES ADDITION ---
    st.write("**Additives (g / 100kg Binder)**")
    fibres_g = st.number_input("Microfibres ", value=0, step=10, min_value=0)

# --- CALCULATION TRIGGER ---

st.markdown("---")

W, msg = solve_mix(t_al_si, t_ca_si, t_h_si,
                   nCa_cem, nSi_fa, nAl_fa, nSi_sf, nAl_sf, nH_water_const,
                   use_fa, use_sf)

if W is None:
    st.error(f"Calculation Failed: {msg}")
else:
    w_cem, w_fa, w_sf, w_theo_water = W

    total_solid_parts = w_cem + w_fa + w_sf
    scale = batch_binder_kg / total_solid_parts

    m_cem = w_cem * scale
    m_fa = w_fa * scale
    m_sf = w_sf * scale
    m_water_req = w_theo_water * scale

    m_sand_dry = batch_binder_kg * sand_binder_ratio

    # Admixture Calcs
    vol_wr_mL = (adm_wr * batch_binder_kg) / 100.0
    vol_vm_mL = (adm_vm * batch_binder_kg) / 100.0
    vol_accel_mL = (adm_accel * batch_binder_kg) / 100.0

    total_adm_water_kg = ((vol_wr_mL + vol_vm_mL + vol_accel_mL) * 1.0) * 0.60 / 1000.0

    # Fibre Calc
    m_fibres_kg = (fibres_g * batch_binder_kg) / 100.0 / 1000.0

    m_sand_wet = m_sand_dry * (1 + (sand_moisture / 100))
    water_from_sand = m_sand_dry * ((sand_moisture - sand_abs) / 100)

    if lock_wb:
        # LOCKED: We reduce added water to maintain the theoretical requirement
        m_water_added = m_water_req - water_from_sand - total_adm_water_kg
        final_water_in_mix = m_water_req
        wb_status = "🔒 Locked (Admix Compensated)"
    else:
        # UNLOCKED: We add the admix ON TOP of the requirement
        m_water_added = m_water_req - water_from_sand
        final_water_in_mix = m_water_req + total_adm_water_kg
        wb_status = "🔓 Unlocked (Water + Admix)"

    real_wb = final_water_in_mix / (m_cem + m_fa + m_sf)

    # PERCENTAGE CALCS
    total_paste_parts = w_cem + w_fa + w_sf + w_theo_water
    pct_cem = (w_cem / total_paste_parts) * 100
    pct_fa = (w_fa / total_paste_parts) * 100
    pct_sf = (w_sf / total_paste_parts) * 100
    pct_water = (w_theo_water / total_paste_parts) * 100

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Effective W/B", f"{real_wb:.3f}", help=wb_status)
    # Update Total Batch Mass to include Fibres
    total_batch_mass = m_cem + m_fa + m_sf + m_sand_wet + max(0, m_water_added) + m_fibres_kg
    r2.metric("Total Batch Mass", f"{total_batch_mass:.2f} kg")
    r3.metric("Fly Ash wt%", f"{pct_fa:.2f}%", help="Based on Total Paste Mass")
    r4.metric("Silica Fume wt%", f"{pct_sf:.2f}%", help="Based on Total Paste Mass")

    if lock_wb:
        st.caption(f"ℹ️ **Status:** {wb_status}. Batch water reduced by **{total_adm_water_kg:.3f} kg**.")
    else:
        st.caption(f"ℹ️ **Status:** {wb_status}. Admix added **{total_adm_water_kg:.3f} kg** extra water.")

    st.success("### 📋 Batch Card")

    if m_water_added < 0:
        st.warning(f"⚠️ **WARNING:** Sand is too wet! You have {abs(m_water_added):.3f} kg excess water.")

    # Add Fibres to DataFrame
    df_data = [
        ["GP Cement", pct_cem, m_cem, "Binder"],
        ["Fly Ash", pct_fa, m_fa, "Binder"],
        ["Silica Fume", pct_sf, m_sf, "Binder"],
        ["Total Water", pct_water, final_water_in_mix, f"Target (Add: {max(0, m_water_added):.3f} kg)"],
        ["Sand (Wet)", "-", m_sand_wet, "Filler"]
    ]

    if m_fibres_kg > 0:
        df_data.append(["Microfibres", "-", m_fibres_kg, "Inert Addition"])

    df = pd.DataFrame(df_data, columns=["Material", "Wt% (Total Paste)", "Mass (kg)", "Notes"])

    st.dataframe(
        df.style.format({
            "Mass (kg)": "{:.3f}",
            "Wt% (Total Paste)": lambda x: "{:.2f}%".format(x) if isinstance(x, (float, int)) else x
        }),
        hide_index=True
    )

    st.info(

        f"**Add Admixtures:** 💧 WR: {vol_wr_mL:.1f} mL | 🍯 VMA: {vol_vm_mL:.1f} mL | ⚡ Accel: {vol_accel_mL:.1f} mL")
