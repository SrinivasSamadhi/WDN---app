import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt
import random

#  Page Configuration 
st.set_page_config(layout="wide", page_title="Water Quality Prediction Dashboard")

#  Sidebar Navigation 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Water Quality Prediction", "Network Visualization"])


EXPECTED_COLUMNS = ['pH', 'Turbidity (NTU)', 'Chlorine (mg/L)']


@st.cache_resource
def load_model_and_scaler(model_path="best_random_forest_model.pkl", scaler_path="wq_scaler.joblib"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

if page == "Water Quality Prediction":
    st.title("💧 Water Contamination Prediction")
    st.write("Enter the water quality parameters to predict contamination.")

    # Load the model and scaler
    model, scaler = load_model_and_scaler()

    if model and scaler:
        st.sidebar.success("Model and scaler loaded successfully!")

        #  Input Fields for Prediction 
        st.header("Input Water Quality Parameters:")

        scaler_min_vals = scaler.data_min_
        scaler_max_vals = scaler.data_max_
        
        feature_ranges = {
            'pH': {'min': 0, 'max': 14.0, 'mean': 7.5},
            'Turbidity (NTU)': {'min': 0.0, 'max': 50.0, 'mean': 2.5},
            'Chlorine (mg/L)': {'min': 0.3, 'max': 5.0, 'mean': 2.5}
        }

 

        input_ph = st.number_input(
            "pH", 
            min_value=float(feature_ranges['pH']['min']), 
            max_value=float(feature_ranges['pH']['max']), 
            value=float(feature_ranges['pH']['mean']), 
            step=0.1
        )
        input_turbidity = st.number_input(
            "Turbidity (NTU)", 
            min_value=float(feature_ranges['Turbidity (NTU)']['min']), 
            max_value=float(feature_ranges['Turbidity (NTU)']['max']), 
            value=float(feature_ranges['Turbidity (NTU)']['mean']), 
            step=0.1
        )
        input_chlorine = st.number_input(
            "Chlorine (mg/L)", 
            min_value=float(feature_ranges['Chlorine (mg/L)']['min']), 
            max_value=float(feature_ranges['Chlorine (mg/L)']['max']), 
            value=float(feature_ranges['Chlorine (mg/L)']['mean']),
            step=0.1
        )

        if st.button("Predict Contamination"):
            # Create a DataFrame for the input, ensuring correct column order
            input_data_dict = {
                'pH': [input_ph],
                'Turbidity (NTU)': [input_turbidity],
                'Chlorine (mg/L)': [input_chlorine]
            }
            input_df = pd.DataFrame(input_data_dict, columns=EXPECTED_COLUMNS) # Enforce order

            try:
                # Scale the input data using the loaded scaler
                input_scaled = scaler.transform(input_df)
                input_df_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)

                # Predict using the loaded model
                prediction = model.predict(input_df_scaled)[0]
                prediction_proba = model.predict_proba(input_df_scaled)[0]

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.error(f"🚨 Prediction: Contaminated")
                    st.metric("Probability of Contamination", f"{prediction_proba[1]:.2%}")
                else:
                    st.success(f"✅ Prediction: Not Contaminated")
                    st.metric("Probability of Not Contaminated", f"{prediction_proba[0]:.2%}")
            
            except ValueError as ve:
                st.error(f"Prediction Error: {ve}. This might be due to an issue with the input data or "
                        f"the model expecting different features. Expected features by scaler: {scaler.feature_names_in_}")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
    else:
        st.warning("Model and/or scaler could not be loaded. Please check the file paths and ensure the files are not corrupted.")
        st.info("You need to save 'rf_model.joblib' and 'wq_scaler.joblib' from your training notebook "
                "into the same directory as this Streamlit app.")

    st.sidebar.markdown("")
    st.sidebar.info("This app uses a pre-trained Random Forest model to predict water contamination.")

elif page == "Network Visualization":
    st.title("📈 Water Distribution Network Visualization")

    dataset = st.selectbox("Select Dataset", ["wdn_synthetic_nodes", "wdn_synthetic_nodes_1", "wdn_synthetic_nodes_2"])
    
    try:
        nodes_df = pd.read_csv(f"networks/{dataset}.csv")
        edges_df = pd.read_csv(f"networks/{dataset.replace('nodes', 'edges')}.csv")

        # Create Graphs
        G_original = nx.Graph()
        for _, row in nodes_df.iterrows():
            G_original.add_node(row["Node_ID"], pos=(row["Longitude"], row["Latitude"]))
        for _, row in edges_df.iterrows():
            G_original.add_edge(row["Start_Node"], row["End_Node"])

        # Damage Network
        G_damaged = G_original.copy()
        edges_to_remove = random.sample(list(G_damaged.edges()), int(0.1 * G_damaged.number_of_edges()))
        G_damaged.remove_edges_from(edges_to_remove)

        # Reconstruct Network
        G_reconstructed = G_damaged.copy()
        G_reconstructed.add_edges_from(edges_to_remove)

        # Plotting
        pos = nx.get_node_attributes(G_original, 'pos')
        fig, axs = plt.subplots(1, 3, figsize=(23, 11))

        nx.draw(G_original, pos, with_labels=True, node_color="skyblue", edge_color="gray", ax=axs[0])
        axs[0].set_title("Original Water Network")

        nx.draw(G_damaged, pos, with_labels=True, node_color="orange", edge_color="gray", ax=axs[1])
        nx.draw_networkx_edges(G_original, pos, edgelist=edges_to_remove, edge_color='red', width=2, ax=axs[1])
        axs[1].set_title("Damaged Network\n(Missing edges in red)")

        nx.draw(G_reconstructed, pos, with_labels=True, node_color="green", edge_color="gray", ax=axs[2])
        nx.draw_networkx_edges(G_reconstructed, pos, edgelist=edges_to_remove, edge_color='blue', width=2, ax=axs[2])
        axs[2].set_title("Reconstructed Network\n(Re-added edges in blue)")

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading or plotting data: {e}")