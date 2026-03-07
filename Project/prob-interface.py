#https://pgmpy.org/
"""
Virtual Lifter: Generative Bayesian Network
-------------------------------------------
This script builds a Bayesian Network based on the Virtual Lifter PRD.
It defines root lifestyle nodes, hidden physiological states, and observable
workout logs, then uses Ancestral Sampling to generate a synthetic dataset.

Dependencies to install before running:
    pip install pgmpy pandas numpy
"""

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling


def _generate_recovery_cpt():
    """Dynamically generate the Recovery CPT based on a simple scoring logic.
    
    Because Recovery has 3 parents with 3 states each, it requires
    3 * 3 * 3 = 27 columns.
    """
    cpt_values = np.zeros((3, 27))
    col = 0
    for sleep in [0, 1, 2]:      # 0: <6, 1: 6-8, 2: >8
        for macros in [0, 1, 2]: # 0: Deficit, 1: Maint, 2: Surplus
            for rest in [0, 1, 2]:   # 0: 0 Days, 1: 1-2 Days, 2: 3+ Days
                score = sleep + macros + rest # Max score = 6, Min = 0
                
                if score <= 2:
                    probs = [0.8, 0.15, 0.05] # Highly likely to be Poor
                elif score <= 4:
                    probs = [0.2, 0.6, 0.2]   # Highly likely to be Adequate
                else:
                    probs = [0.05, 0.15, 0.8] # Highly likely to be Optimal
                
                cpt_values[:, col] = probs
                col += 1
    return cpt_values.tolist()


def build_model():
    """Construct and validate the Virtual Lifter Bayesian Network.
    
    Returns:
        DiscreteBayesianNetwork: The validated model with all CPDs attached.
    """
    model = DiscreteBayesianNetwork([
        # Layer 1 to Layer 2
        ('Sleep', 'Recovery'),
        ('Macros', 'Recovery'),
        ('TimeRest', 'Recovery'),
        ('TimeRest', 'Fatigue'),
        
        # Layer 2 Internal Dependencies
        ('Recovery', 'Readiness'),
        ('Fatigue', 'Readiness'),
        ('Fatigue', 'Risk'),
        ('Recovery', 'Soreness'),
        
        # Layer 2 to Layer 3 (The Log Outputs)
        ('Readiness', 'Weight'),
        ('Readiness', 'Volume'),
        ('Risk', 'Volume'),
        ('Weight', 'RPE'),
        ('Volume', 'RPE')
    ])

    # We use state_names so our final generated CSV contains readable text (e.g., "Surplus") 
    # instead of integer codes (e.g., 2).

    cpd_sleep = TabularCPD(variable='Sleep', variable_card=3,
                           values=[[0.3], [0.5], [0.2]], # <6, 6-8, >8
                           state_names={'Sleep': ['<6 hours', '6-8 hours', '>8 hours']})

    cpd_macros = TabularCPD(variable='Macros', variable_card=3,
                            values=[[0.3], [0.4], [0.3]], # Deficit, Maint, Surplus
                            state_names={'Macros': ['Deficit', 'Maintenance', 'Surplus']})

    cpd_timerest = TabularCPD(variable='TimeRest', variable_card=3,
                              values=[[0.2], [0.5], [0.3]], # 0 Days, 1-2 Days, 3+ Days
                              state_names={'TimeRest': ['0 Days', '1-2 Days', '3+ Days']})


    # --- LAYER 2: HIDDEN NODES ---

    # Fatigue (Depends on TimeRest)
    # Columns represent TimeRest: [0 Days, 1-2 Days, 3+ Days]
    cpd_fatigue = TabularCPD(
        variable='Fatigue', variable_card=3,
        evidence=['TimeRest'], evidence_card=[3],
        values=[
            [0.1, 0.5, 0.8], # Low Fatigue
            [0.3, 0.4, 0.15], # Moderate Fatigue
            [0.6, 0.1, 0.05]  # High Fatigue
        ],
        state_names={'Fatigue': ['Low', 'Moderate', 'High'],
                     'TimeRest': ['0 Days', '1-2 Days', '3+ Days']}
    )

    # Injury Risk (Depends on Fatigue)
    # Columns represent Fatigue: [Low, Moderate, High]
    cpd_risk = TabularCPD(
        variable='Risk', variable_card=3,
        evidence=['Fatigue'], evidence_card=[3],
        values=[
            [0.8, 0.4, 0.1], # Low Risk
            [0.15, 0.4, 0.3], # Elevated Risk
            [0.05, 0.2, 0.6]  # Critical Risk
        ],
        state_names={'Risk': ['Low', 'Elevated', 'Critical'],
                     'Fatigue': ['Low', 'Moderate', 'High']}
    )

    # Recovery (Depends on Sleep, Macros, TimeRest)
    cpd_recovery = TabularCPD(
        variable='Recovery', variable_card=3,
        evidence=['Sleep', 'Macros', 'TimeRest'], evidence_card=[3, 3, 3],
        values=_generate_recovery_cpt(),
        state_names={'Recovery': ['Poor', 'Adequate', 'Optimal'],
                     'Sleep': ['<6 hours', '6-8 hours', '>8 hours'],
                     'Macros': ['Deficit', 'Maintenance', 'Surplus'],
                     'TimeRest': ['0 Days', '1-2 Days', '3+ Days']}
    )

    # Readiness (Depends on Recovery, Fatigue)
    # 9 columns. Parents: Recovery (Poor, Adq, Opt), Fatigue (Low, Mod, High)
    cpd_readiness = TabularCPD(
        variable='Readiness', variable_card=3,
        evidence=['Recovery', 'Fatigue'], evidence_card=[3, 3],
        # Col Order: (Poor,Low), (Poor,Mod), (Poor,High), (Adq,Low), (Adq,Mod), (Adq,High), (Opt,Low), (Opt,Mod), (Opt,High)
        values=[
            [0.6, 0.8, 0.95,  0.2, 0.4, 0.7,   0.05, 0.2, 0.5], # Low Readiness
            [0.3, 0.15, 0.04,  0.6, 0.5, 0.2,   0.25, 0.6, 0.4], # Medium Readiness
            [0.1, 0.05, 0.01,  0.2, 0.1, 0.1,   0.70, 0.2, 0.1]  # High Readiness
        ],
        state_names={'Readiness': ['Low', 'Medium', 'High'],
                     'Recovery': ['Poor', 'Adequate', 'Optimal'],
                     'Fatigue': ['Low', 'Moderate', 'High']}
    )

    # --- LAYER 3: LEAF NODES (Outputs) ---

    # Soreness (Depends on Recovery)
    cpd_soreness = TabularCPD(
        variable='Soreness', variable_card=3,
        evidence=['Recovery'], evidence_card=[3],
        values=[
            [0.1, 0.4, 0.8], # None
            [0.3, 0.5, 0.15], # Mild
            [0.6, 0.1, 0.05]  # Severe
        ],
        state_names={'Soreness': ['None', 'Mild', 'Severe'],
                     'Recovery': ['Poor', 'Adequate', 'Optimal']}
    )

    # Weight Lifted (Depends on Readiness)
    cpd_weight = TabularCPD(
        variable='Weight', variable_card=3,
        evidence=['Readiness'], evidence_card=[3],
        values=[
            [0.7, 0.2, 0.05], # Light
            [0.2, 0.6, 0.25], # Working
            [0.1, 0.2, 0.7]   # Heavy
        ],
        state_names={'Weight': ['Light (<70% 1RM)', 'Working (70-85% 1RM)', 'Heavy (>85% 1RM)'],
                     'Readiness': ['Low', 'Medium', 'High']}
    )

    # Volume (Depends on Readiness, Risk)
    # 9 columns. Parents: Readiness (Low, Med, High), Risk (Low, Elev, Crit)
    cpd_volume = TabularCPD(
        variable='Volume', variable_card=3,
        evidence=['Readiness', 'Risk'], evidence_card=[3, 3],
        values=[
            [0.4, 0.6, 0.9,  0.2, 0.4, 0.8,  0.1, 0.3, 0.7], # Low Volume
            [0.4, 0.3, 0.05, 0.6, 0.5, 0.15, 0.3, 0.5, 0.2], # Target Volume
            [0.2, 0.1, 0.05, 0.2, 0.1, 0.05, 0.6, 0.2, 0.1]  # High Volume
        ],
        state_names={'Volume': ['Low', 'Target', 'High'],
                     'Readiness': ['Low', 'Medium', 'High'],
                     'Risk': ['Low', 'Elevated', 'Critical']}
    )

    # RPE (Depends on Weight, Volume)
    # 9 columns. Parents: Weight (Light, Work, Heavy), Volume (Low, Targ, High)
    cpd_rpe = TabularCPD(
        variable='RPE', variable_card=3,
        evidence=['Weight', 'Volume'], evidence_card=[3, 3],
        values=[
            [0.8, 0.6, 0.4,  0.6, 0.3, 0.1,  0.3, 0.1, 0.01], # RPE 6-7
            [0.15, 0.3, 0.4, 0.3, 0.5, 0.5,  0.5, 0.5, 0.2],  # RPE 8-9
            [0.05, 0.1, 0.2, 0.1, 0.2, 0.4,  0.2, 0.4, 0.79]  # RPE 10
        ],
        state_names={'RPE': ['RPE 6-7 (Easy)', 'RPE 8-9 (Hard)', 'RPE 10 (Max)'],
                     'Weight': ['Light (<70% 1RM)', 'Working (70-85% 1RM)', 'Heavy (>85% 1RM)'],
                     'Volume': ['Low', 'Target', 'High']}
    )

    model.add_cpds(cpd_sleep, cpd_macros, cpd_timerest, 
                   cpd_fatigue, cpd_risk, cpd_recovery, 
                   cpd_readiness, cpd_soreness, cpd_weight, 
                   cpd_volume, cpd_rpe)

    assert model.check_model() == True
    return model


def generate_samples(model, num_samples=10000):
    """Generate synthetic lifting logs via ancestral (forward) sampling.
    
    Args:
        model: A validated DiscreteBayesianNetwork.
        num_samples: Number of synthetic rows to generate.
    
    Returns:
        pd.DataFrame: Synthetic dataset ordered from causes to effects.
    """
    simulator = BayesianModelSampling(model)
    synthetic_data = simulator.forward_sample(size=num_samples, show_progress=True)

    # Reorder columns so they read logically from cause to effect
    column_order = [
        'Sleep', 'Macros', 'TimeRest',          # Roots
        'Recovery', 'Fatigue', 'Readiness', 'Risk', # Hidden
        'Soreness', 'Weight', 'Volume', 'RPE'   # Outputs
    ]
    return synthetic_data[column_order]


# ── Run when executed directly (preserves original behavior) ──
if __name__ == "__main__":
    print("Building the Virtual Lifter DAG...")
    model = build_model()
    print("Model validation successful! All probabilities sum to 1.\n")

    num_samples = 10000
    print(f"Running simulation to generate {num_samples} lifting logs...")
    synthetic_data = generate_samples(model, num_samples)

    output_filename = "synthetic_lifting_logs.csv"
    synthetic_data.to_csv(output_filename, index=False)

    print(f"\nSuccess! Generated data saved to '{output_filename}'.")
    print("\nData Preview (First 5 Rows):")
    print(synthetic_data.head())