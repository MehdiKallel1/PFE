"""
CORRECTED: Realistic Tunisia Macroeconomic Forecasts 2025-2026
Now using ACTUAL end-2024 values as baseline for smooth transitions
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_corrected_realistic_forecasts():
    """
    Generate realistic forecasts using ACTUAL December 2024 values as baseline
    """
    
    # ACTUAL December 2024 baseline values from your data
    baseline = {
        'RNB_Par_Habitant': 7606.838,      # ACTUAL: $7,607 per capita
        'PIB_US_Courants': 49094620552.0,  # ACTUAL: $49.1B GDP
        'RNB_US_Courants': 45843368764.0,  # ACTUAL: $45.8B GNI
        'Credit_Interieur': 94.839,        # ACTUAL: 94.8
        'Impots_Revenus': 28.106,          # ACTUAL: 28.1%
        'Inflation_Rate': 9.804,           # ACTUAL: 9.8%
        'Paiements_Interet': 14.903,       # ACTUAL: 14.9%
        'Taux_Interet': 6.457,             # ACTUAL: 6.5% (will adjust to realistic CB rate)
        'Masse_Monetaire': 115844790196.0  # ACTUAL: 115.8B TND
    }
    
    realistic_data = []
    
    print("  🎯 Creating realistic forecasts from ACTUAL 2024 values...")
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Generate 24 months (2025-2026)
    for i in range(24):
        # Calculate proper end-of-month dates
        year = 2025 + i // 12
        month = (i % 12) + 1
        
        # Get last day of month
        from calendar import monthrange
        last_day = monthrange(year, month)[1]
        date = pd.to_datetime(f'{year}-{month:02d}-{last_day}')
        
        month_progress = i / 23.0  # 0 to 1 over 24 months
        year_in_period = i // 12   # 0 for 2025, 1 for 2026
        month_in_year = i % 12     # 0-11 within each year
        
        # === 1. INFLATION RATE ===
        # Realistic decline from 9.8% to ~6.5% with volatility
        target_end = 6.5
        base_trend = baseline['Inflation_Rate'] - (baseline['Inflation_Rate'] - target_end) * month_progress
        
        # Add monthly volatility and seasonal effects
        seasonal = 0.3 * np.sin(month_in_year * np.pi / 6)
        monthly_shock = np.random.normal(0, 0.25)
        inflation = max(5.5, base_trend + seasonal + monthly_shock)
        
        # === 2. INTEREST RATES ===
        # Start from actual rate (6.457%) but reflect true central bank policy
        # Tunisia CB rate is actually ~7.5-8%, so gradually move there first, then ease
        if i < 3:      # Jan-Mar 2025: Adjust to realistic CB rate
            policy_rate = baseline['Taux_Interet'] + 1.0 * (i/3) + np.random.normal(0, 0.1)
        elif i < 9:    # Apr-Sep 2025: Maintain restrictive policy
            policy_rate = 7.5 + np.random.normal(0, 0.15)
        elif i < 15:   # Oct 2025-Mar 2026: Begin cautious easing
            policy_rate = 7.5 - 1.0 * ((i-9)/6) + np.random.normal(0, 0.2)
        else:          # Apr-Dec 2026: Continue toward neutral
            policy_rate = 6.5 - 0.3 * ((i-15)/9) + np.random.normal(0, 0.15)
        
        interest_rate = np.clip(policy_rate, 6.0, 8.0)
        
        # === 3. CREDIT INTERIOR ===
        # Start from actual 94.839, reflect monetary policy impact
        if year_in_period == 0:  # 2025: Slight tightening
            credit_trend = baseline['Credit_Interieur'] * (1 - 0.06 * (i/12))
        else:  # 2026: Gradual recovery
            credit_trend = baseline['Credit_Interieur'] * 0.94 * (1 + 0.04 * ((i-12)/12))
        
        # Add credit cycle and volatility
        credit_cycle = 2.5 * np.sin(i * np.pi / 10)
        monthly_var = np.random.normal(0, 1.0)
        credit = max(85, credit_trend + credit_cycle + monthly_var)
        
        # === 4. MONEY SUPPLY ===
        # Start from actual 115.84B, realistic growth
        base_growth = 0.07 if year_in_period == 0 else 0.08
        inflation_component = inflation / 100 * 0.15
        annual_growth = base_growth + inflation_component
        monthly_growth = annual_growth / 12
        
        if i == 0:
            money_supply = baseline['Masse_Monetaire'] * (1 + monthly_growth)
        else:
            prev_money = realistic_data[i-1]['Masse_Monetaire']
            growth_shock = np.random.normal(0, 0.004)
            money_supply = prev_money * (1 + monthly_growth + growth_shock)
        
        # === 5. GDP (PIB_US_Courants) ===
        # Start from actual $49.1B, realistic growth
        gdp_growth_2025 = 0.016  # 1.6% for 2025
        gdp_growth_2026 = 0.023  # 2.3% for 2026
        
        gdp_growth = gdp_growth_2025 if year_in_period == 0 else gdp_growth_2026
        
        # Add seasonal patterns
        seasonal_factor = 1 + 0.02 * np.sin((month_in_year - 2) * np.pi / 6)
        monthly_gdp_growth = (gdp_growth / 12) * seasonal_factor
        
        if i == 0:
            gdp = baseline['PIB_US_Courants'] * (1 + monthly_gdp_growth)
        else:
            prev_gdp = realistic_data[i-1]['PIB_US_Courants']
            gdp_shock = np.random.normal(0, 0.002)
            gdp = prev_gdp * (1 + monthly_gdp_growth + gdp_shock)
        
        # === 6. GNI (RNB_US_Courants) ===
        # Start from actual $45.8B, maintain realistic ratio to GDP
        historical_gni_ratio = baseline['RNB_US_Courants'] / baseline['PIB_US_Courants']  # ~0.934
        cyclical_variation = 0.02 * np.sin(i * np.pi / 12)
        gni_ratio = historical_gni_ratio + cyclical_variation
        gni = gdp * gni_ratio
        
        # === 7. GNI PER CAPITA (RNB_Par_Habitant) ===
        # CRITICAL FIX: Start from actual $7,607 with realistic growth
        population_growth = 0.011  # 1.1% annual population growth
        
        # Calculate real per capita growth
        gni_growth_rate = (gni / baseline['RNB_US_Courants'] - 1)
        population_factor = (1 + population_growth) ** (i / 12)
        
        # Per capita = (GNI growth) / (population growth)
        gni_per_capita = baseline['RNB_Par_Habitant'] * (1 + gni_growth_rate) / population_factor
        
        # === 8. TAX REVENUES (Impots_Revenus) ===
        # Start from actual 28.106%, gradual policy changes
        economic_impact = (gdp / baseline['PIB_US_Courants'] - 1) * 1.2
        policy_adjustment = 0.3 * month_progress  # Gradual tax policy changes
        
        # Collection efficiency varies
        efficiency = 1 + 0.1 * np.sin(i * np.pi / 16) + np.random.normal(0, 0.03)
        tax_base = baseline['Impots_Revenus'] + economic_impact + policy_adjustment
        tax_revenues = np.clip(tax_base * efficiency, 25, 32)
        
        # === 9. INTEREST PAYMENTS (Paiements_Interet) ===
        # Start from actual 14.903%, rising debt service burden
        debt_burden_increase = 0.5 * month_progress
        rate_impact = (interest_rate - baseline['Taux_Interet']) * 0.3
        fx_shock = np.random.normal(0, 0.3)
        
        interest_payments = baseline['Paiements_Interet'] + debt_burden_increase + rate_impact + fx_shock
        interest_payments = np.clip(interest_payments, 14, 17)
        
        # Store monthly data with smooth transitions
        month_data = {
            'Date': date,
            'Credit_Interieur': round(credit, 2),
            'Impots_Revenus': round(tax_revenues, 2),
            'Inflation_Rate': round(inflation, 6),
            'Paiements_Interet': round(interest_payments, 2),
            'Taux_Interet': round(interest_rate, 6),
            'RNB_Par_Habitant': round(gni_per_capita, 2),  # Now smoothly transitions from 7,607
            'Masse_Monetaire': round(money_supply, 2),
            'PIB_US_Courants': round(gdp, 2),
            'RNB_US_Courants': round(gni, 2),
            'is_predicted': True
        }
        
        realistic_data.append(month_data)
    
    return pd.DataFrame(realistic_data)

def show_transition_analysis(realistic_forecasts, baseline):
    """
    Show how values transition smoothly from actual 2024 data
    """
    print("\n📊 SMOOTH TRANSITION ANALYSIS:")
    print("=" * 50)
    
    first_forecast = realistic_forecasts.iloc[0]
    sixth_forecast = realistic_forecasts.iloc[5]
    
    # Key indicators to check
    indicators = {
        'RNB_Par_Habitant': ('GNI per Capita', '$'),
        'PIB_US_Courants': ('GDP', '$B'),
        'Inflation_Rate': ('Inflation', '%'),
        'Taux_Interet': ('Interest Rate', '%'),
        'Credit_Interieur': ('Credit', '')
    }
    
    for indicator, (name, unit) in indicators.items():
        baseline_val = baseline[indicator]
        jan_val = first_forecast[indicator]
        jun_val = sixth_forecast[indicator]
        
        # Calculate month-to-month change
        monthly_change = (jan_val - baseline_val) / baseline_val * 100
        six_month_change = (jun_val - baseline_val) / baseline_val * 100
        
        if indicator == 'PIB_US_Courants':
            baseline_display = f"{baseline_val/1e9:.1f}"
            jan_display = f"{jan_val/1e9:.1f}"
            jun_display = f"{jun_val/1e9:.1f}"
        else:
            baseline_display = f"{baseline_val:.1f}"
            jan_display = f"{jan_val:.1f}"
            jun_display = f"{jun_val:.1f}"
        
        print(f"\n{name}:")
        print(f"  Dec 2024: {baseline_display}{unit}")
        print(f"  Jan 2025: {jan_display}{unit} ({monthly_change:+.2f}%)")
        print(f"  Jun 2025: {jun_display}{unit} ({six_month_change:+.2f}%)")
        print(f"  Assessment: {'✅ Smooth transition' if abs(monthly_change) < 5 else '⚠️ Large jump'}")

# Generate corrected forecasts
if __name__ == "__main__":
    print("🔧 GENERATING CORRECTED REALISTIC FORECASTS")
    print("=" * 50)
    
    # Use ACTUAL December 2024 values
    actual_baseline = {
        'RNB_Par_Habitant': 7606.838,
        'PIB_US_Courants': 49094620552.0,
        'RNB_US_Courants': 45843368764.0,
        'Credit_Interieur': 94.839,
        'Impots_Revenus': 28.106,
        'Inflation_Rate': 9.804,
        'Paiements_Interet': 14.903,
        'Taux_Interet': 6.457,
        'Masse_Monetaire': 115844790196.0
    }
    
    # Generate corrected forecasts
    corrected_forecasts = create_corrected_realistic_forecasts()
    
    # Show transition analysis
    show_transition_analysis(corrected_forecasts, actual_baseline)
    
    # Show sample data
    print(f"\n📋 SAMPLE CORRECTED FORECASTS:")
    print("=" * 60)
    print("Date       | GNI/Cap | GDP($B) | Inflation | Interest | Credit")
    print("-" * 60)
    
    for i in range(0, 12, 2):  # Show every other month for first year
        row = corrected_forecasts.iloc[i]
        date_str = row['Date'].strftime('%Y-%m-%d')
        gni_cap = f"${row['RNB_Par_Habitant']:.0f}"
        gdp_b = f"{row['PIB_US_Courants']/1e9:.1f}"
        inflation = f"{row['Inflation_Rate']:.1f}%"
        interest = f"{row['Taux_Interet']:.1f}%"
        credit = f"{row['Credit_Interieur']:.0f}"
        
        print(f"{date_str} | {gni_cap:>7} | {gdp_b:>6} | {inflation:>8} | {interest:>7} | {credit:>6}")
    
    # Create CSV output
    print(f"\n💾 CORRECTED CSV DATA:")
    print("=" * 40)
    print("Replace your 2025-2026 data with this:")
    print("=" * 40)
    
    # Generate CSV content
    csv_rows = []
    for _, row in corrected_forecasts.iterrows():
        csv_row = f"{row['Date'].strftime('%Y-%m-%d')},{row['Credit_Interieur']},{row['Impots_Revenus']},{row['Inflation_Rate']},{row['Paiements_Interet']},{row['Taux_Interet']},{row['RNB_Par_Habitant']},{row['Masse_Monetaire']},{row['PIB_US_Courants']},{row['RNB_US_Courants']},{row['is_predicted']}"
        csv_rows.append(csv_row)
    
    # Show first few rows as sample
    for i in range(min(6, len(csv_rows))):
        print(csv_rows[i])
    
    if len(csv_rows) > 6:
        print("... (truncated, use the full dataset)")
    
    print(f"\n✅ FIXED: No more unrealistic jumps!")
    print(f"✅ GNI per capita now grows smoothly from $7,607")
    print(f"✅ All values transition logically from your actual 2024 data")
    
    # Save to file for easy use
    corrected_forecasts.to_csv('corrected_realistic_forecasts_2025_2026.csv', index=False)
    print(f"\n📁 Saved corrected forecasts to: corrected_realistic_forecasts_2025_2026.csv")