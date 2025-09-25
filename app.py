import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import json
import re
from io import BytesIO
import random

# --- MASTER PROMPT (Version 4.0 - Hardcoded into the App) ---
MASTER_PROMPT = """
### ROLE & GOAL ###
You are "Athena," a highly skilled Organisational Psychologist and Assessment Designer. Your expertise lies in creating psychometrically sound, behavioral Situational Judgement Tests (SJTs). Your mission is to generate plausible and nuanced SJTs for a professional, hierarchical environment like the UAE Ministry of Defence, based on a provided competency framework.

### GUIDING PRINCIPLES & CONSTRAINTS (NON-NEGOTIABLE) ###
1. **Behavioral, Not Technical:** Situations and response options must be purely behavioral. They must not require any specialized technical knowledge or domain-specific jargon to understand or answer.
2. **Generic Roles:** Scenarios should be written for a general "individual contributor," "team member," or "junior officer" level. Avoid using specific, high-ranking roles.
3. **Distinct Contexts:** Each SJT you generate for a given competency must have a unique and distinct context. Do not repeat scenarios (e.g., if one SJT is about a project deadline, the next should be about a team conflict or a new procedure).
4. **Plausible Distractors:** The incorrect options must be plausible, realistic actions that a reasonable person might consider, not obviously foolish or malicious choices.

### CORE KNOWLEDGE: COMPETENCY FRAMEWORK ###
You will be provided with a competency and a list of its behavioral indicators for the PL1 (Proficiency Level 1).

<competency_data>
<name>{{COMPETENCY_NAME}}</name>
<definition>{{COMPETENCY_DEFINITION}}</definition>
<pl1_indicators>
<indicator>{{INDICATOR_1}}</indicator>
<indicator>{{INDICATOR_2}}</indicator>
<indicator>{{INDICATOR_3}}</indicator>
<indicator>{{INDICATOR_4}}</indicator>
</pl1_indicators>
</competency_data>

### EXEMPLARY CASE (FEW-SHOT LEARNING) ###
To ensure you understand the required style and quality, study these SME-approved examples. Note how the situations create a dilemma between different professional behaviors and how the options are scored on a four-point scale.

<sjt_gold_standard_example>
<competency>Effective Intelligence</competency>
<situation>You are leading your squad during officer training in preparing for a readiness assessment tomorrow. The assessment will involve a map-reading test, a timed equipment drill, and a short tactical briefing to instructors. You only have one evening to prepare, and the squad is tired after a long day. It is unlikely that you can prepare equally well for all three components. How would you approach this situation?</situation>
<options>
<option score="0.75">Split the squad evenly across the three components so that all areas receive some attention, even if this reduces the depth of preparation for all the tasks overall.</option>
<option score="0.25">Start preparation with the component that seems easiest for the squad to handle, reasoning that building early confidence will make the team more motivated for the harder tasks.</option>
<option score="0.0">Follow the order of the assessment as given (map-reading, then equipment, then briefing), preparing in that sequence without changing the plan.</option>
<option score="1.0">Quickly assess which component the squad is weakest on and focus the majority of your limited time on that area to prevent a critical failure, even if it means other areas are only briefly reviewed.</option>
</options>
</sjt_gold_standard_example>

<sjt_gold_standard_example>
<competency>Communication</competency>
<situation>You are preparing a short progress update for a group of stakeholders from different departments. Some are interested in technical details, while others only want a clear overview of the key outcomes. You have limited time to present, and you need to make sure everyone understands the information. How would you navigate this situation?</situation>
<options>
<option score="0.75">Share the update verbally with both detail and overview included, aiming to cover as much ground as possible within the time available.</option>
<option score="0.0">Share all the detailed points in the order they occurred, even if it means the update runs longer than expected.</option>
<option score="0.25">Focus your update mainly on the aspects you are most familiar with, even if it does not fully address what each audience expects.</option>
<option score="1.0">Share a concise overview supported by simple visuals, highlight the main outcomes, and provide technical details in a handout for those who want more depth.</option>
</options>
</sjt_gold_standard_example>

### COGNITIVE WORKFLOW: TASK ###
Your task is to generate a new, unique SJT for the competency provided in the `CORE KNOWLEDGE` section. Follow these steps meticulously:

1. **Select Two Target Indicators:** From the list of four indicators in `<pl1_indicators>`, choose **two** that can be put into a state of tension or conflict. These two indicators will be the focus of your scenario.
2. **Brainstorm a Dilemma:** Create a realistic work scenario that forces a person to choose between behaviors related to the two indicators you selected.
3. **Write the Situation & Question:** Draft a clear, concise paragraph (70-100 words) describing the dilemma, followed by a question like "What is your most effective course of action?"
4. **Generate Four Behavioral Options:** Create four distinct, plausible behavioral options.
   * **One option must be the "Positive High" (ideal) response (maps to score 1.0).** This option often synthesizes the best aspects of both target indicators.
   * **The other three options should be plausible but suboptimal distractors, mapping to "Positive Low" (0.75), "Negative Low" (0.25), and "Negative High" (0.0).**
5. **Provide Rationale:** In a separate block, briefly explain why you chose the two target indicators and how the situation creates a dilemma for them.

### OUTPUT FORMAT ###
You MUST provide the output in a single, clean JSON object. Do not include any text outside of the JSON structure.

```json
{
  "competency_name": "{{COMPETENCY_NAME}}",
  "sjt_indicator_mapping": [
    "{{THE FIRST INDICATOR YOU CHOSE}}",
    "{{THE SECOND INDICATOR YOU CHOSE}}"
  ],
  "sjt": {
    "situation": "Your generated situation text here.",
    "question": "Your generated question here.",
    "options": [
      { "option_text": "Action A text here.", "score_level": "Positive High" },
      { "option_text": "Action B text here.", "score_level": "Positive Low" },
      { "option_text": "Action C text here.", "score_level": "Negative Low" },
      { "option_text": "Action D text here.", "score_level": "Negative High" }
    ]
  },
  "validation_rationale": "A brief explanation of the dilemma created between the two chosen indicators."
}
```
"""

# --- App Configuration ---
st.set_page_config(page_title="Athena: SJT Generation Studio", page_icon="🤖", layout="wide")

# --- Helper Functions ---
def get_api_key():
    """Retrieves the Gemini API key from Streamlit secrets."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()

def parse_gemini_response(text):
    """Safely parses the JSON response from the Gemini model."""
    try:
        # Handle potential markdown code block wrapping the JSON
        if text.strip().startswith("```json"):
            text = text.strip()[7:-3]
        return json.loads(text)
    except json.JSONDecodeError:
        st.warning("Failed to decode JSON from model response. Skipping this item.")
        return None

def parse_indicators(indicator_blob):
    """Parses a string of newline-separated indicators into a clean list."""
    if not isinstance(indicator_blob, str):
        return []
    # Split by any newline character and filter out empty strings
    indicators = [ind.strip() for ind in re.split(r'\r\n|\r|\n', indicator_blob) if ind.strip()]
    return indicators

# --- Main Application UI ---
st.title("Athena 🤖: SJT Generation Studio")
st.markdown("### For the UAE Ministry of Defence Competency Framework")
st.markdown("---")

st.info(
    "**Instructions:**\n"
    "1. Upload your competency matrix Excel file (must contain 'Competency Name', 'Theme', and 'Indicators (PL1)' columns).\n"
    "2. Click the 'Generate SJTs' button.\n"
    "3. The application will generate **5 unique SJTs** for each competency in your file.\n"
    "4. Review the results and download the final Excel file."
)

uploaded_file = st.file_uploader("Upload Your Competency Matrix (.xlsx)", type="xlsx")

if uploaded_file:
    try:
        df_competencies_raw = pd.read_excel(uploaded_file)
        required_columns = {'Competency Name', 'Theme', 'Indicators (PL1)'}

        if not required_columns.issubset(df_competencies_raw.columns):
            st.error(f"File is missing required columns. Please ensure your file has: {', '.join(required_columns)}")
        else:
            # --- DATA PREPROCESSING & CLEANING STEP ---
            # Forward fill to handle merged-cell-like structure in Excel
            df_competencies_raw['Competency Name'].ffill(inplace=True)
            df_competencies_raw['Theme'].ffill(inplace=True)

            # Drop rows where indicators are still NaN after filling
            df_competencies_raw.dropna(subset=['Indicators (PL1)'], inplace=True)

            # Group by competency and aggregate the indicators into a single string
            df_competencies = df_competencies_raw.groupby(['Competency Name', 'Theme'])['Indicators (PL1)'].apply('\n'.join).reset_index()

            st.success(f"✅ File uploaded and processed! Found {len(df_competencies)} valid competencies to process.")
            st.dataframe(df_competencies[['Competency Name', 'Theme']].head())

            if st.button("🚀 Generate 5 SJTs per Competency", type="primary", use_container_width=True):
                if df_competencies.empty:
                    st.error("No valid competencies found in the uploaded file after processing. Please check your file and try again.")
                else:
                    api_key = get_api_key()
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-pro-latest')

                    all_results = []
                    total_generations = len(df_competencies) * 5
                    progress_bar = st.progress(0, text="Initializing generation...")

                    with st.spinner("Athena is crafting behavioral dilemmas... This may take a few minutes."):
                        for i, row in df_competencies.iterrows():
                            competency_name = row['Competency Name']
                            competency_def = row['Theme']
                            # The 'Indicators (PL1)' column now contains the full, aggregated string
                            pl1_indicators = parse_indicators(row['Indicators (PL1)'])

                            if len(pl1_indicators) < 2:
                                st.warning(f"Skipping '{competency_name}' as it has fewer than 2 indicators after processing.")
                                continue

                            for j in range(5):
                                progress_index = (i * 5 + j)
                                progress_bar.progress(progress_index / total_generations, text=f"Generating SJT {j+1}/5 for: '{competency_name}'")

                                # Prepare the prompt for the API call
                                prompt = MASTER_PROMPT.replace("{{COMPETENCY_NAME}}", competency_name)
                                prompt = prompt.replace("{{COMPETENCY_DEFINITION}}", competency_def)
                                
                                # Pad with empty strings if there are fewer than 4 indicators
                                padded_indicators = pl1_indicators + [''] * (4 - len(pl1_indicators))
                                prompt = prompt.replace("{{INDICATOR_1}}", padded_indicators[0])
                                prompt = prompt.replace("{{INDICATOR_2}}", padded_indicators[1])
                                prompt = prompt.replace("{{INDICATOR_3}}", padded_indicators[2])
                                prompt = prompt.replace("{{INDICATOR_4}}", padded_indicators[3])

                                try:
                                    response = model.generate_content(prompt)
                                    parsed_data = parse_gemini_response(response.text)

                                    if parsed_data:
                                        # Flatten the JSON response for the output DataFrame
                                        flat_data = {
                                            "Competency": parsed_data.get("competency_name"),
                                            "SJT Number": f"SJT {j+1}",
                                            "Indicator Mapping 1": parsed_data.get("sjt_indicator_mapping", ["", ""])[0],
                                            "Indicator Mapping 2": parsed_data.get("sjt_indicator_mapping", ["", ""])[1],
                                            "Situation": parsed_data.get("sjt", {}).get("situation"),
                                            "Question": parsed_data.get("sjt", {}).get("question"),
                                            "Rationale": parsed_data.get("validation_rationale")
                                        }
                                        options = parsed_data.get("sjt", {}).get("options", [])
                                        score_map = {"Positive High": 0, "Positive Low": 1, "Negative Low": 2, "Negative High": 3}
                                        # Sort options by score level to ensure consistent column order
                                        sorted_options = sorted(options, key=lambda x: score_map.get(x.get('score_level'), 99))

                                        for k, option in enumerate(sorted_options):
                                            flat_data[f"Option {k+1} Text"] = option.get("option_text")
                                            flat_data[f"Option {k+1} Score Level"] = option.get("score_level")
                                        
                                        all_results.append(flat_data)
                                    
                                    time.sleep(1) # Add a small delay to avoid hitting rate limits
                                except Exception as e:
                                    st.error(f"An API error occurred for '{competency_name}': {e}")
                                    continue

                    progress_bar.progress(1.0, text="Generation complete!")

                    if all_results:
                        st.success("🎉 Generation finished successfully!")
                        df_results = pd.DataFrame(all_results)
                        st.dataframe(df_results)

                        # Create an in-memory Excel file for download
                        output_excel = BytesIO()
                        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                            df_results.to_excel(writer, index=False, sheet_name='Generated_SJTs')
                        
                        st.download_button(
                            label="💾 Download Generated_MOD_SJTs.xlsx",
                            data=output_excel.getvalue(),
                            file_name="Generated_MOD_SJTs.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    else:
                        st.error("Generation finished, but no SJTs were successfully created. Please check the warnings above and verify your input file's format and content.")

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        st.error("Please ensure it's a valid Excel (.xlsx) file and the column names are correct.")

