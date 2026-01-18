# Medical Disclaimer & System Limitations
# إخلاء المسؤولية الطبية وحدود النظام

## 1. General Disclaimer
**This software is for research and educational purposes only.**
The content, models, and predictions provided by this repository are not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or calculated using this system.

## 2. Limitations of "Ground Truth" (Labeling Methodology)
Users must be aware that the **Fatty Liver Disease (NAFLD)** classification in this dataset was generated using a **Rule-Based Approach**, not Histopathology (Biopsy).
* [cite_start]**Method:** The target labels (Healthy vs. Sick) were inferred based on biochemical thresholds derived from clinical guidelines[cite: 109, 110].
* **Thresholds Used:**

    * [cite_start]**Triglycerides > 150 mg/dL**: Based on NCEP ATP III standards for metabolic syndrome[cite: 111].
    * [cite_start]**GGT > 40 IU/L**: Used as a sensitive marker for oxidative stress[cite: 117].
* **Implication:** While these rules are medically sound for screening, they do not replace the "Gold Standard" of liver biopsy.

## 3. Data Imputation & The "Veto" Safety Net
In scenarios where user input is incomplete, the system may impute "Normal" value# Medical Disclaimer & System Limitations
# إخلاء المسؤولية الطبية وحدود النظام

## 1. General Disclaimer
**This software is for research and educational purposes only.**
The content, models, and predictions provided by this repository are not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or calculated using this system.

## 2. Limitations of "Ground Truth" (Labeling Methodology)
Users must be aware that the **Fatty Liver Disease (NAFLD)** classification in this dataset was generated using a **Rule-Based Approach**, not Histopathology (Biopsy).
* [cite_start]**Method:** The target labels (Healthy vs. Sick) were inferred based on biochemical thresholds derived from clinical guidelines[cite: 109, 110].
* **Thresholds Used:**
    * [cite_start]**ALT > 40 IU/L**: Based on ACG Clinical Guidelines to indicate liver injury[cite: 113, 114].
    * [cite_start]**Triglycerides > 150 mg/dL**: Based on NCEP ATP III standards for metabolic syndrome[cite: 111].
    * [cite_start]**GGT > 40 IU/L**: Used as a sensitive marker for oxidative stress[cite: 117].
* **Implication:** While these rules are medically sound for screening, they do not replace the "Gold Standard" of liver biopsy.

## 3. Data Imputation & The "Veto" Safety Net
In scenarios where user input is incomplete, the system may impute "Normal" values to facilitate processing.
* [cite_start]**Risk:** This imputation can theoretically lead to **False Negatives** in the Donor Eligibility Model.
* [cite_start]**Mitigation:** A "Veto System" has been implemented to override favorable donor predictions if the separate **Fibrosis Model** detects high-risk stages (Stage 2-4) using independent markers (Platelets/Prothrombin)[cite: 21, 22]. However, no safety system is fail-safe.

## 4. Applicability
This model is calibrated using specific demographic data and may not generalize perfectly to populations with significantly different genetic or environmental profiles.isk:** This imputation can theoretically lead to **False Negatives*.
 However, no safety system is fail-safe.

## 4. Applicability
This model is calibrated using specific demographic data and may not generalize perfectly to populations with significantly different genetic or environmental profiles.
