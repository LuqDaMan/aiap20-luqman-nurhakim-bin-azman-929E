# src/app/main_streamlit.py
import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
# Assuming src is in PYTHONPATH or the execution is relative to the project root
try:
    from src.utils.stream_utils import (
        STREAMLIT_LOGGER as logger, # Use a shorter alias
        DEPLOY_CONFIG,
        call_fastapi_predict
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up to project root
    from src.utils.stream_utils import (
        STREAMLIT_LOGGER as logger,
        DEPLOY_CONFIG,
        call_fastapi_predict
    )

# --- Page Configuration (FR-UI-001) ---
APP_CONFIG = DEPLOY_CONFIG.get("streamlit_app", {})
PAGE_TITLE = APP_CONFIG.get("title", "Bank Term Deposit Prediction")
st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide", # Or "centered"
    initial_sidebar_state="expanded" # Or "collapsed"
)

logger.info(f"Streamlit application '{PAGE_TITLE}' started.")

# --- Load necessary configurations ---
RAW_FEATURES_CONFIG = APP_CONFIG.get("raw_features_form_input", [])
AVAILABLE_MODELS_CONFIG = DEPLOY_CONFIG.get("available_models", {})

if not RAW_FEATURES_CONFIG:
    st.error("Critical Error: Raw features configuration ('raw_features_form_input') is missing from deploy_config.yaml.")
    logger.critical("raw_features_form_input missing from deploy_config.yaml. Streamlit app cannot render form.")
    st.stop() # Stop execution if core config is missing

if not AVAILABLE_MODELS_CONFIG:
    st.error("Critical Error: Available models configuration ('available_models') is missing from deploy_config.yaml.")
    logger.critical("available_models missing from deploy_config.yaml. Streamlit app cannot render model selection.")
    st.stop()


# --- Helper function to render input widgets dynamically ---
def render_widget(feature_config: Dict[str, Any], current_values: Dict[str, Any]) -> Any:
    """Renders a Streamlit widget based on feature configuration."""
    widget_type = feature_config.get("widget")
    name = feature_config.get("name")
    label = feature_config.get("label", name.replace("_", " ").title())
    params = feature_config.get("params", {})
    
    # Get current value for the widget if available (e.g. for pdays interaction)
    # This helps if one widget's value depends on another during reruns before submission
    current_value = current_values.get(name, params.get("value", params.get("index", None)))
    
    # Update params with current value if not already set by 'value' or 'index' from config
    if 'value' not in params and 'index' not in params and current_value is not None:
        if widget_type in ["selectbox", "radio"] and 'options' in params:
            try:
                params['index'] = params['options'].index(current_value)
            except ValueError: # If current_value not in options, use default
                if 'index' not in params: # check if index already exists
                    params['index'] = 0 
        else:
            params['value'] = current_value

    widget_map = {
        "number_input": st.number_input,
        "selectbox": st.selectbox,
        "radio": st.radio,
        "text_input": st.text_input,
        "checkbox": st.checkbox,
        # Add other widget types if needed
    }

    if widget_type in widget_map:
        # For selectbox/radio, if params['value'] is provided, it might conflict with params['index']
        # Streamlit prefers 'index' for these if 'value' is one of the options.
        # We will rely on the 'index' or 'value' being correctly set in params from config.
        if widget_type in ["selectbox", "radio"] and 'value' in params:
            # If 'value' is provided in params, try to convert it to an index
            # This makes config more flexible if one prefers to set default by value string
            if 'options' in params and params['value'] in params['options']:
                try:
                    params['index'] = params['options'].index(params['value'])
                    del params['value'] # Avoid conflict
                except ValueError:
                    logger.warning(f"Default value '{params['value']}' for {label} not in options. Using default index.")
                    if 'index' not in params: params['index'] = 0 # Default to first option
            elif 'index' not in params: # if value cannot be mapped and no index, default index
                 params['index'] = 0


        # Special handling for pdays toggle as discussed by user
        if name == "previously_contacted_toggle":
            # This is the checkbox. Its value will be used to influence pdays.
            # The actual 'pdays' input field might be shown/hidden or its value set by this.
            # For simplicity, we will use this toggle to determine the pdays value directly.
            # The original 'pdays' field from config might not be directly rendered if this toggle handles it.
            # Or, this toggle's state could be used to set the value of a separate 'pdays' field.
             return widget_map[widget_type](label=label, key=f"st_{name}", **params)
        elif name == "pdays" and any(f.get('name') == 'previously_contacted_toggle' for f in RAW_FEATURES_CONFIG):
            # If 'previously_contacted_toggle' exists, 'pdays' behavior is linked.
            # The user's note: "set days to 0 if checked, 999 if unchecked" for the toggle.
            # We'll handle this logic when collecting form data.
            # Here, we just render the pdays input as configured. It might be informational or allow override.
            # For now, render as usual, logic applied on submit.
            return widget_map[widget_type](label=label, key=f"st_{name}", **params)
        else:
            return widget_map[widget_type](label=label, key=f"st_{name}", **params)
    else:
        logger.warning(f"Unsupported widget type '{widget_type}' for feature '{name}'. Defaulting to text_input.")
        return st.text_input(label=label, value=params.get("value", ""), key=f"st_{name}", help=params.get("help", ""))


# --- Main UI Layout ---
st.title(PAGE_TITLE)
st.markdown("Enter client details and select a model to predict term deposit subscription.")

# Using columns for layout if desired
# col1, col2 = st.columns([2,1]) # Input form in col1, results in col2

# Input Form (FR-UI-002, FR-UI-004)
with st.form(key="prediction_form"):
    st.subheader("Client Information & Model Selection")

    # Store current form values, useful for interdependent widgets if form reruns before submission
    # Not strictly necessary for this simple case yet, but good practice for complex forms.
    # For now, Streamlit's widget state handles this.
    form_values: Dict[str, Any] = {}

    # Model Selection (FR-UI-003)
    model_display_names = {key: config.get("name_display", key.replace("_", " ").title())
                           for key, config in AVAILABLE_MODELS_CONFIG.items()}
    selected_model_display_name = st.selectbox(
        label="1. Select Prediction Model",
        options=list(model_display_names.values()),
        key="st_model_selection",
        help="Choose the machine learning model for prediction."
    )
    # Get the actual model key from the display name
    selected_model_key = next(
        key for key, name in model_display_names.items() if name == selected_model_display_name
    )
    form_values["selected_model_key"] = selected_model_key # Store for submission

    st.markdown("---")
    st.markdown("##### 2. Enter Client Raw Data Features:")
    
    # Dynamically generate input fields based on config (FR-UI-002a)
    # Create two columns for inputs for better layout
    input_col1, input_col2 = st.columns(2)
    
    # Handle previously_contacted_toggle first if it exists
    toggle_config = next((f for f in RAW_FEATURES_CONFIG if f.get('name') == 'previously_contacted_toggle'), None)
    pdays_config = next((f for f in RAW_FEATURES_CONFIG if f.get('name') == 'pdays'), None)

    # Render toggle in the first column if it exists
    if toggle_config:
        with input_col1:
            form_values[toggle_config['name']] = render_widget(toggle_config, form_values)
    
    # Render other features
    feature_idx = 0
    for feature_conf in RAW_FEATURES_CONFIG:
        # Skip toggle if already rendered, skip pdays if toggle will fully control it (decision point)
        if feature_conf['name'] == 'previously_contacted_toggle':
            continue
        
        # If the toggle sets pdays, we might not need a pdays input field.
        # User's note: toggle "will set days to 0 if checked, 999 if unchecked".
        # So, `pdays` input field may not be needed or should be read-only based on toggle.
        # For now, let's assume toggle's value is primary for pdays.
        # We can conditionally show pdays input or make it disabled later if needed.
        # The 'pdays' field in client_data sent to backend WILL be set by the toggle.
        # If 'pdays' is in RAW_FEATURES_CONFIG, it means it's a feature the API expects.
        # So, we will still pass 'pdays'. The toggle will determine its value.

        # Distribute other inputs into columns
        current_column = input_col1 if feature_idx % 2 == 0 else input_col2
        with current_column:
            form_values[feature_conf['name']] = render_widget(feature_conf, form_values)
        feature_idx += 1

    # Submit button for the form
    submitted = st.form_submit_button("🚀 Get Prediction")


# --- Form Submission Logic & Displaying Results ---
if submitted:
    logger.info(f"Form submitted. Selected model: {selected_model_key}")
    
    client_data_payload: Dict[str, Any] = {}
    # Collect data from form_values (which are Streamlit widget outputs)
    for feature_conf in RAW_FEATURES_CONFIG:
        feature_name = feature_conf['name']
        if feature_name == 'previously_contacted_toggle': # Skip the toggle itself as an input feature
            continue 
        
        # Get value from the form_values dict, which holds widget outputs
        # st.session_state could also be used, but direct widget return is simpler for forms
        client_data_payload[feature_name] = form_values.get(feature_name)

    # Handle the 'previously_contacted_toggle' logic to set 'pdays' value
    # User's change request: `deploy_config.yaml` has `previously_contacted_toggle` with this logic:
    # "Check this if the client was previously contacted (will set days to 0 if checked, 999 if unchecked)."
    if toggle_config and pdays_config: # Ensure both toggle and pdays config exist
        is_previously_contacted = form_values.get(toggle_config['name'], False) # Default to False if not found
        if is_previously_contacted:
            client_data_payload[pdays_config['name']] = 0 # Set pdays to 0 if toggle is checked
            logger.debug(f"'{toggle_config['name']}' is checked. Setting '{pdays_config['name']}' to 0.")
        else:
            client_data_payload[pdays_config['name']] = 999 # Set pdays to 999 if toggle is unchecked
            logger.debug(f"'{toggle_config['name']}' is unchecked. Setting '{pdays_config['name']}' to 999.")
    elif pdays_config and pdays_config['name'] not in client_data_payload : # if toggle not present but pdays is
         client_data_payload[pdays_config['name']] = form_values.get(pdays_config['name']) # use direct pdays input

    # Log the final client data being sent
    logger.debug(f"Client data for prediction: {client_data_payload}")

    # Basic client-side validation example (can be expanded)
    if not all(client_data_payload.get(f['name']) is not None for f in RAW_FEATURES_CONFIG if f['name'] != 'previously_contacted_toggle'):
        st.error("Please fill in all required client data fields.")
        logger.warning("Submission attempt with missing fields.")
    else:
        # Show processing message (FR-UI-006)
        with st.spinner(f"Processing with model: {model_display_names[selected_model_key]}... Please wait."):
            prediction_result, error_msg = call_fastapi_predict(
                model_name=selected_model_key,
                client_data_dict=client_data_payload
            )

        if error_msg:
            st.error(f"Prediction Failed: {error_msg}")
            logger.error(f"Error received from FastAPI for model '{selected_model_key}': {error_msg}")
        elif prediction_result:
            logger.info(f"Prediction successful: {prediction_result}")
            st.success("🎉 Prediction Complete!") # (FR-UI-006)
            
            # Display prediction results clearly (FR-UI-005)
            st.subheader("Prediction Result:")
            
            # Using columns for a nicer layout of results
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="Model Used", value=prediction_result.get("model_used", "N/A"))
            with res_col2:
                predicted_cls = prediction_result.get("predicted_class", "N/A").title()
                st.metric(label="Predicted Subscription", value=predicted_cls)
            with res_col3:
                prob_yes = prediction_result.get("probability_yes")
                if prob_yes is not None:
                    st.metric(label="Probability (Yes)", value=f"{prob_yes:.2%}")
                else:
                    st.metric(label="Probability (Yes)", value="N/A")

            with st.expander("View Raw JSON Response from API"):
                st.json(prediction_result)
            with st.expander("View Input Data Sent to API"):
                st.json({"model_name_sent": selected_model_key, "client_data_sent": client_data_payload})
        else:
            # Should ideally be caught by error_msg, but as a fallback
            st.error("An unknown error occurred. No prediction data received.")
            logger.error("Unknown error: call_fastapi_predict returned (None, None).")

# --- Footer or additional information ---
st.markdown("---")
st.caption(f"© {datetime.now().year} {PAGE_TITLE}. Version based on PRD_II v1.1 ({DEPLOY_CONFIG.get('api',{}).get('version','N/A')})")

logger.info("Streamlit application rendering complete.")