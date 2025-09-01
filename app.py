import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import json
from io import BytesIO

# --- MASTER PROMPT (Version 2.0) ---
MASTER_PROMPT = """
### ROLE & GOAL ###
You are "Athena," a highly skilled Organisational Psychologist and Military Scenario Designer. You are a subject matter expert in competency-based assessment and have deep knowledge of command structures, operational terminology, and the cultural context of the UAE Armed Forces. Your mission is to adapt validated corporate Situational Judgement Tests (SJTs) into realistic, culturally relevant military scenarios for the UAE Ministry of Defence, while strictly preserving the underlying psychometric framework.
**Your identity as "Athena" is non-negotiable. Maintain this expert persona consistently.**

### CORE KNOWLEDGE: COMPETENCY FRAMEWORK ###
You will be provided with a specific competency to work with. Internalize its definition and behavioral indicators.

<competency_definition>
  <name>{{COMPETENCY_NAME}}</name>
  <definition>{{COMPETENCY_DEFINITION}}</definition>
  <indicators>
    <positive_high>{{INDICATOR_POSITIVE_HIGH}}</positive_high>
    <positive_low>{{INDICATOR_POSITIVE_LOW}}</positive_low>
    <negative_low>{{INDICATOR_NEGATIVE_LOW}}</negative_low>
    <negative_high>{{INDICATOR_NEGATIVE_HIGH}}</negative_high>
  </indicators>
</competency_definition>

### CONTEXT: DOMAIN ADAPTATION ###
- **Source Domain:** Standard corporate office environment.
- **Target Domain:** UAE Ministry of Defence. 
- **Key Cultural & Operational Factors:**
    - **Hierarchy & Chain of Command:** Decisions are made within a strict hierarchical structure. Respect for authority is paramount.
    - **Discipline & Procedure:** Actions are often governed by Standard Operating Procedures (SOPs).
    - **High-Stakes Environment:** Decisions can have significant consequences for safety, mission success, and national security.
    - **Team Cohesion:** Strong emphasis on teamwork ("unit," "squad," "platoon").
- **Scenario Sub-domains (Choose one per generation):** Logistics & Supply Chain, Strategic Planning, Inter-branch Coordination (Joint Operations), Personnel Management & Training, Base Administration, or Military Intelligence Analysis.
- **UAE Strategic Themes:** Emphasize themes of national service, technological advancement (a key UAE priority), and international collaboration, reflecting the UAE's strategic position.
- **Realism Mandate:** Scenarios should reflect the professional, day-to-day challenges of a modern, technologically advanced military, NOT cinematic combat situations or Hollywood tropes.

### EXEMPLARY CASE (FEW-SHOT LEARNING) ###
<sjt_adaptation_example>
  <competency>Agility</competency>
  <source_sjt context="Corporate">
    <situation>Your organization has recently implemented a new performance management system. Most of your team members are finding it complicated and are struggling to complete their reviews. You are also more familiar and comfortable with the previous system.</situation>
    <question>What is the most effective action for you to take?</question>
    <options>
        <option indicator="Positive High">Volunteer to be a 'super-user' for the new system to help your colleagues.</option>
        <option indicator="Positive Low">Wait for the formal training sessions to be scheduled by the HR department.</option>
        <option indicator="Negative High">Continue using the templates from the old system to submit your team's reviews.</option>
        <option indicator="Negative Low">Attend the required training but express skepticism about the system's benefits to your colleagues.</option>
    </options>
  </source_sjt>
  <target_sjt context="UAE Ministry of Defence">
    <situation>Command has mandated the rollout of a new digital Command and Control (C2) platform for mission planning, replacing the traditional map-and-marker-based system. Many junior officers in your unit are finding the new platform non-intuitive and are slow to adopt it during training exercises. You are also more proficient with the previous, manual methods.</situation>
    <question>What is the most effective action for you to take?</question>
    <options>
        <option indicator="Positive High">Proactively spend your off-duty time mastering the platform, then volunteer to run informal training sessions for your peers to help the unit improve its readiness.</option>
        <option indicator="Positive Low">Follow the official training schedule provided and trust that the unit's proficiency will improve over time as instructed.</option>
        <option indicator="Negative High">During planning exercises, continue to rely solely on the old manual system, arguing it's more reliable under pressure.</option>
        <option indicator="Negative Low">Attend the mandatory training sessions but openly state in debriefs that the old system was faster and less prone to technical failure.</option>
    </options>
  </target_sjt>
</sjt_adaptation_example>

### COGNITIVE WORKFLOW: TASK ###
Your task is to generate a completely new and unique SJT (Situation and 4 Options) for the provided competency, tailored for the UAE Ministry of Defence. Follow these steps meticulously:
1.  **Deconstruct the Competency:** Re-read the `competency_definition` and its behavioral indicators.
2.  **Select a Sub-domain:** Choose one of the specified `Scenario Sub-domains` to frame your situation.
3.  **Brainstorm a Scenario:** Conceive a realistic, challenging, and plausible non-combat scenario relevant to a junior officer within the chosen sub-domain.
4.  **Write the Situation:** Draft a clear and concise paragraph describing the situation (70-100 words).
5.  **Formulate the Question:** Write a clear question asking what the most effective (or appropriate) course of action is.
6.  **Generate Four Options:** Create four distinct, plausible options. Each option MUST be a behavioral manifestation of one of the four indicators. **Crucially, the three distractor options must be *plausible and tempting* actions.**
7.  **Provide Rationale:** In a separate reasoning block, explicitly state which option maps to which behavioral indicator and why.

### OUTPUT FORMAT & CONSTRAINTS ###
- **Format:** You MUST provide the output in a single, clean JSON object. Do not include any text outside of the JSON structure.
- **Tone:** The language must be formal, precise, and use appropriate military terminology.
- **Realism:** The scenario must be believable within the specified context.
- **Final Check:** Before outputting the JSON, perform a final self-critique. Does the scenario realistically test the competency? Are the options distinct and plausible? Does the rationale clearly link the options to the indicators? If not, revise before finalizing.
```json
{
  "competency_name": "{{COMPETENCY_NAME}}",
  "sjt": {
    "situation": "Your generated situation text here.",
    "question": "Your generated question here.",
    "options": [
      { "option_text": "Action A text here.", "indicator_mapping": "positive_high" },
      { "option_text": "Action B text here.", "indicator_mapping": "positive_low" },
      { "option_text": "Action C text here.", "indicator_mapping": "negative_low" },
      { "option_text": "Action D text here.", "indicator_mapping": "negative_high" }
    ]
  },
  "validation_rationale": {
    "positive_high_rationale": "Explain why this option demonstrates the highest level of the competency.",
    "positive_low_rationale": "Explain why this option is positive but suboptimal.",
    "negative_low_rationale": "Explain why this option demonstrates a minor negative behavior.",
    "negative_high_rationale": "Explain why this option demonstrates a significant negative behavior."
  }
}
```
"""

# --- App Configuration ---
st.set_page_config(
    page_title="Athena: SJT Generation Studio",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---

def create_sample_excel():
    """Generates an in-memory sample Excel file for download."""
    sample_data = {
        'Competency': ['Agility', 'Strategic Thinking'],
        'Definition': [
            'The ability to adapt to new situations, embrace change, and remain effective in a constantly evolving environment.',
            'The ability to see the bigger picture, understand the long-term implications of actions, and align tactical decisions with overarching goals.'
        ],
        'Positive High': [
            'Proactively seeks out and champions new ways of working.',
            'Analyzes situations from a broad, long-term perspective before acting.'
        ],
        'Positive Low': [
            'Accepts and follows new procedures when mandated.',
            'Considers immediate consequences but may miss long-term effects.'
        ],
        'Negative Low': [
            'Expresses skepticism about change but eventually complies.',
            'Focuses only on completing the immediate task without considering strategic context.'
        ],
        'Negative High': [
            'Actively resists change and continues to use old methods.',
            'Acts impulsively, potentially compromising strategic objectives.'
        ]
    }
    df_sample = pd.DataFrame(sample_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_sample.to_excel(writer, index=False, sheet_name='Competencies')
    processed_data = output.getvalue()
    return processed_data

def get_api_key():
    """Fetches the Gemini API key from Streamlit secrets."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()

def parse_gemini_response(text):
    """Safely parses the JSON string from the Gemini response."""
    try:
        # The model sometimes wraps the JSON in ```json ... ```, so we clean it.
        if text.strip().startswith("```json"):
            text = text.strip()[7:-3]
        return json.loads(text)
    except json.JSONDecodeError:
        return None # Return None if parsing fails

# --- Main Application UI ---

st.title("Athena 🤖: SJT Generation Studio")
st.markdown("### For the UAE Ministry of Defence")
st.markdown("---")

# --- Step 1: Instructions & Sample Download ---
st.header("Step 1: Get the Correct Input Format")
st.markdown(
    "Download the sample Excel file to see the required format. "
    "Your uploaded file must contain these exact column headers: `Competency`, `Definition`, "
    "`Positive High`, `Positive Low`, `Negative Low`, `Negative High`."
)
st.download_button(
    label="📥 Download Sample_Competencies.xlsx",
    data=create_sample_excel(),
    file_name="Sample_Competencies.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --- Step 2: File Upload ---
st.header("Step 2: Upload Your Competencies File")
uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type="xlsx"
)

if uploaded_file:
    try:
        df_competencies = pd.read_excel(uploaded_file)
        # --- Validation ---
        required_columns = {'Competency', 'Definition', 'Positive High', 'Positive Low', 'Negative Low', 'Negative High'}
        if not required_columns.issubset(df_competencies.columns):
            st.error(f"File is missing one or more required columns. Please ensure you have: {', '.join(required_columns)}")
        else:
            st.success(f"✅ File uploaded successfully! Found {len(df_competencies)} competencies.")
            st.dataframe(df_competencies)

            # --- Step 3: Generation ---
            st.header("Step 3: Generate SJTs")
            if st.button("🚀 Generate 3 SJTs per Competency", type="primary"):
                # Configure API
                api_key = get_api_key()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-pro')
                
                all_results = []
                total_generations = len(df_competencies) * 3
                progress_bar = st.progress(0, text="Initializing generation...")

                with st.spinner("Athena is generating scenarios... This may take a few minutes."):
                    for i, row in df_competencies.iterrows():
                        for j in range(3): # Generate 3 SJTs per competency
                            progress_text = f"Generating SJT {j+1}/3 for competency: '{row['Competency']}'"
                            progress_bar.progress((i * 3 + j) / total_generations, text=progress_text)
                            
                            # Populate the prompt
                            prompt = MASTER_PROMPT.replace("{{COMPETENCY_NAME}}", str(row['Competency']))
                            prompt = prompt.replace("{{COMPETENCY_DEFINITION}}", str(row['Definition']))
                            prompt = prompt.replace("{{INDICATOR_POSITIVE_HIGH}}", str(row['Positive High']))
                            prompt = prompt.replace("{{INDICATOR_POSITIVE_LOW}}", str(row['Positive Low']))
                            prompt = prompt.replace("{{INDICATOR_NEGATIVE_LOW}}", str(row['Negative Low']))
                            prompt = prompt.replace("{{INDICATOR_NEGATIVE_HIGH}}", str(row['Negative High']))
                            
                            try:
                                response = model.generate_content(prompt)
                                parsed_data = parse_gemini_response(response.text)
                                
                                if parsed_data:
                                    # Flatten the JSON into a dictionary
                                    flat_data = {
                                        "Competency": parsed_data.get("competency_name"),
                                        "SJT Number": f"SJT {j+1}",
                                        "Situation": parsed_data.get("sjt", {}).get("situation"),
                                        "Question": parsed_data.get("sjt", {}).get("question")
                                    }
                                    # Add options
                                    options = parsed_data.get("sjt", {}).get("options", [])
                                    for k, option in enumerate(options):
                                        flat_data[f"Option {k+1} Text"] = option.get("option_text")
                                        flat_data[f"Option {k+1} Indicator"] = option.get("indicator_mapping")
                                    all_results.append(flat_data)
                                else:
                                    st.warning(f"Could not parse response for '{row['Competency']}' SJT {j+1}. Skipping.")

                                time.sleep(1) # Small delay to avoid hitting rate limits

                            except Exception as e:
                                st.error(f"An API error occurred for '{row['Competency']}': {e}")
                                continue

                progress_bar.progress(1.0, text="Generation complete!")
                
                if all_results:
                    st.success("🎉 Generation finished successfully!")
                    df_results = pd.DataFrame(all_results)
                    st.dataframe(df_results)
                    
                    # --- Download Results ---
                    output_excel = BytesIO()
                    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                        df_results.to_excel(writer, index=False, sheet_name='Generated_SJTs')
                    
                    st.download_button(
                        label="💾 Download Generated_MOD_SJTs.xlsx",
                        data=output_excel.getvalue(),
                        file_name="Generated_MOD_SJTs.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
