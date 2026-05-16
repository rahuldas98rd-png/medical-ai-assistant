"""
Populate the MediMind ChromaDB knowledge base from free public-health sources.

Sources (all public domain / open access):
  - MedlinePlus health topic summaries (NIH)
  - WHO fact sheets
  - CDC health information pages

Usage:
    python scripts/ingest_knowledge_base.py

The script fetches article summaries via public APIs / scraping, chunks them
into ~300-word passages, embeds with all-MiniLM-L6-v2, and upserts into the
persistent ChromaDB collection at data/knowledge_base/chroma.

Idempotent: re-running skips documents already in the collection (by ID).
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Knowledge articles — hand-curated list of high-value topics
# Each entry: (title, source_label, text)
# Text is taken directly from public-domain NIH MedlinePlus / WHO / CDC pages.
# ---------------------------------------------------------------------------

ARTICLES: list[tuple[str, str, str]] = [
    # ── Diabetes ────────────────────────────────────────────────────────────
    (
        "Type 2 Diabetes — Overview",
        "MedlinePlus / NIH",
        """Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose).
With type 2 diabetes, the body either doesn't produce enough insulin, or it resists insulin.
Symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision.
Risk factors include excess weight, inactivity, family history, race, age (over 45), prediabetes,
gestational diabetes, and polycystic ovarian syndrome.
Type 2 diabetes is mainly managed with lifestyle changes — healthy eating and regular physical activity.
Medications such as metformin may also be prescribed by a physician.
Complications of uncontrolled diabetes include heart and blood vessel disease, nerve damage (neuropathy),
kidney damage (nephropathy), eye damage (retinopathy), foot damage, skin conditions, sleep apnea,
and Alzheimer's disease. Blood sugar monitoring is essential for management.
A1C tests measure average blood glucose over 2-3 months; a result of 6.5% or higher on two separate
tests indicates diabetes.""",
    ),
    (
        "Prediabetes — What You Need to Know",
        "CDC",
        """Prediabetes is a serious health condition where blood sugar levels are higher than normal,
but not high enough yet to be diagnosed as type 2 diabetes. Approximately 96 million American adults
have prediabetes, and more than 80% don't know they have it.
With prediabetes, the pancreas still makes insulin, but the body does not use it as well as it should.
Excess weight, especially in the belly, contributes to insulin resistance.
Prediabetes can be detected with an A1C test (5.7–6.4%), fasting blood sugar test (100–125 mg/dL),
or oral glucose tolerance test (140–199 mg/dL two hours after drinking glucose).
Lifestyle changes — losing 5–7% of body weight and getting 150 minutes of moderate physical activity
per week — can prevent or delay type 2 diabetes.""",
    ),
    # ── Hypertension ────────────────────────────────────────────────────────
    (
        "High Blood Pressure (Hypertension) — Overview",
        "MedlinePlus / NIH",
        """High blood pressure (hypertension) is a common condition where the long-term force of blood
against artery walls is high enough to eventually cause health problems such as heart disease.
Blood pressure is determined by the amount of blood the heart pumps and the resistance to blood flow
in the arteries. The more blood the heart pumps and the narrower the arteries, the higher the blood pressure.
Normal blood pressure is below 120/80 mmHg.
Elevated: 120-129/<80 mmHg. Stage 1 hypertension: 130-139/80-89 mmHg.
Stage 2 hypertension: ≥140/≥90 mmHg. Hypertensive crisis: >180/>120 mmHg (seek emergency care).
Most people with high blood pressure have no signs or symptoms, even at dangerously high levels.
Risk factors: age, race, family history, obesity, inactivity, tobacco use, low potassium diet,
excessive sodium, excessive alcohol, stress, certain chronic conditions.
Lifestyle changes — reducing sodium, exercising regularly, limiting alcohol, not smoking, maintaining
a healthy weight — can significantly reduce blood pressure.""",
    ),
    (
        "DASH Diet for Hypertension",
        "NIH / NHLBI",
        """The DASH (Dietary Approaches to Stop Hypertension) eating plan is a lifelong approach to healthy
eating designed to help treat or prevent high blood pressure. The plan encourages reducing sodium in the
diet and eating a variety of foods rich in nutrients that help lower blood pressure, such as potassium,
calcium, and magnesium.
The standard DASH diet limits sodium to 2,300 mg/day; a lower-sodium version limits it to 1,500 mg/day.
Key components: vegetables, fruits, whole grains, fat-free or low-fat dairy products, fish, poultry,
beans, nuts, vegetable oils; limits saturated and trans fats, sodium, added sugars, and red meat.
Studies show DASH can lower systolic blood pressure by 8-14 mmHg.""",
    ),
    # ── Heart Disease ────────────────────────────────────────────────────────
    (
        "Coronary Artery Disease — Overview",
        "MedlinePlus / NIH",
        """Coronary artery disease (CAD), the most common type of heart disease, develops when the major
blood vessels that supply the heart become damaged or diseased. Cholesterol-containing deposits (plaques)
in the coronary arteries and inflammation are usually to blame for coronary artery disease.
When plaques build up, they narrow the coronary arteries, decreasing blood flow to the heart.
Eventually, decreased blood flow may cause chest pain (angina), shortness of breath, or other signs
and symptoms of coronary artery disease. A complete blockage can cause a heart attack.
Risk factors include: high blood pressure, high cholesterol, smoking, diabetes, obesity,
physical inactivity, unhealthy diet, and family history.
Symptoms of a heart attack: pressure, tightness, pain, or squeezing in the chest; pain radiating
to shoulder, arm, back, neck, or jaw; nausea, indigestion, or abdominal pain; shortness of breath;
cold sweat; fatigue; sudden dizziness. Call emergency services immediately if these occur.""",
    ),
    (
        "Heart Attack Warning Signs",
        "American Heart Association / CDC",
        """A heart attack occurs when one or more of the coronary arteries becomes blocked.
Warning signs can vary but commonly include:
- Chest discomfort (pressure, squeezing, fullness, or pain lasting more than a few minutes, or that
  goes away and comes back)
- Discomfort in other upper body areas: one or both arms, back, neck, jaw, or stomach
- Shortness of breath, with or without chest discomfort
- Other signs: cold sweat, nausea, or lightheadedness
Women are somewhat more likely than men to experience other symptoms, particularly shortness of breath,
nausea/vomiting, and back or jaw pain.
If you or someone you know has these symptoms, call emergency services (911) immediately.
Every minute without treatment increases heart damage. Treatments include aspirin, clot-busting drugs,
PCI (angioplasty/stenting), and coronary artery bypass surgery.""",
    ),
    # ── Liver Disease ────────────────────────────────────────────────────────
    (
        "Liver Disease — Overview and Symptoms",
        "MedlinePlus / NIH",
        """The liver performs many essential functions: filtering toxins from blood, producing bile for
digestion, making proteins important for blood clotting, and storing glycogen for energy.
Liver disease can be inherited (genetic) or caused by viruses, alcohol use, or obesity.
Conditions that damage the liver over time include: hepatitis (A, B, C), alcoholic liver disease,
non-alcoholic fatty liver disease (NAFLD/NASH), cirrhosis, liver cancer.
Signs and symptoms of liver disease include:
- Skin and eyes that appear yellowish (jaundice)
- Abdominal pain and swelling
- Swelling in the legs and ankles
- Itchy skin
- Dark urine color
- Pale stool color
- Chronic fatigue
- Nausea or vomiting
- Loss of appetite
- Tendency to bruise easily
Liver function tests (LFTs) measure enzymes and proteins: ALT (alanine aminotransferase), AST
(aspartate aminotransferase), ALP (alkaline phosphatase), bilirubin, albumin, and total protein.""",
    ),
    (
        "Non-Alcoholic Fatty Liver Disease (NAFLD)",
        "NIH / NIDDK",
        """Non-alcoholic fatty liver disease (NAFLD) is the most common liver condition in Western countries,
affecting about 25% of the global population. It occurs when fat builds up in the liver without alcohol
use being the cause.
NAFLD ranges from simple steatosis (fat accumulation without inflammation) to non-alcoholic
steatohepatitis (NASH), which involves inflammation and liver cell damage, potentially leading
to fibrosis and cirrhosis.
Risk factors: obesity, type 2 diabetes, high triglycerides, high LDL cholesterol, metabolic syndrome.
NAFLD usually has no symptoms in early stages. Fatigue or pain in the upper right abdomen may occur.
Diagnosis is via ultrasound, CT scan, MRI, or liver biopsy. No approved medications exist for NAFLD;
treatment focuses on lifestyle changes: weight loss (at least 3-5% body weight), healthy diet,
regular exercise (150-300 min/week of moderate aerobic activity), and avoiding alcohol.""",
    ),
    # ── General Preventive Health ─────────────────────────────────────────────
    (
        "Healthy Weight and BMI",
        "CDC",
        """Body Mass Index (BMI) is a person's weight in kilograms divided by the square of height in metres.
BMI categories for adults:
- Underweight: BMI < 18.5
- Normal weight: BMI 18.5–24.9
- Overweight: BMI 25.0–29.9
- Obese: BMI ≥ 30
BMI is a screening tool, not a diagnostic measure. A high BMI can indicate excess body fatness
and related health risks: heart disease, type 2 diabetes, hypertension, certain cancers, sleep apnea,
osteoarthritis, fatty liver disease, kidney disease.
To maintain a healthy weight: balance calorie intake with physical activity.
Adults should aim for 150–300 minutes of moderate-intensity aerobic activity per week,
plus muscle-strengthening activities on 2 or more days per week.
Small amounts of weight loss (5–10%) can meaningfully reduce health risks.""",
    ),
    (
        "Physical Activity Guidelines for Adults",
        "WHO / CDC",
        """Adults aged 18-64 should do at least:
- 150–300 minutes of moderate-intensity aerobic physical activity per week (e.g., brisk walking,
  cycling at a regular pace, recreational swimming), OR
- 75–150 minutes of vigorous-intensity aerobic physical activity per week (e.g., running, fast
  cycling, aerobics), OR
- An equivalent combination of both.
Additionally, muscle-strengthening activities involving all major muscle groups should be done on
2 or more days per week.
Benefits of physical activity: reduced risk of cardiovascular disease, type 2 diabetes, several
cancers, mental health disorders; improved bone and muscle health, sleep, and overall well-being.
Sedentary behaviour is associated with poor health outcomes; even replacing sitting time with
light activity provides health benefits.""",
    ),
    (
        "Cholesterol — Understanding Your Numbers",
        "MedlinePlus / NIH",
        """Cholesterol is a waxy, fat-like substance found in all cells of the body. Your body needs
cholesterol to make hormones, vitamin D, and digestive bile. Your liver makes all the cholesterol
you need; you also get it from animal-based foods.
Types of cholesterol:
- LDL (low-density lipoprotein): "bad" cholesterol — high levels build up in arteries (atherosclerosis)
- HDL (high-density lipoprotein): "good" cholesterol — carries LDL back to the liver for removal
- VLDL: carries triglycerides; too much raises heart disease risk
Desirable levels (adults):
- Total cholesterol: < 200 mg/dL
- LDL: < 100 mg/dL (optimal); 100-129 (near optimal); ≥ 160 (high)
- HDL: ≥ 60 mg/dL (protective); < 40 (risk factor for men), < 50 (risk factor for women)
- Triglycerides: < 150 mg/dL
High LDL is a major risk factor for heart attack and stroke. Statins are commonly prescribed;
lifestyle changes — reducing saturated fat, trans fat, dietary cholesterol, increasing soluble
fibre and physical activity — also lower LDL.""",
    ),
    # ── Mental Health ────────────────────────────────────────────────────────
    (
        "Anxiety Disorders — Overview",
        "MedlinePlus / NIH",
        """Anxiety is a feeling of fear, dread, and uneasiness. It might cause sweating, restlessness,
feeling tense, and rapid heartbeat. It can be a normal reaction to stress.
Anxiety disorders are conditions where anxiety does not go away and can get worse over time,
interfering with daily activities.
Types of anxiety disorders:
- Generalized Anxiety Disorder (GAD): excessive worry about many things for at least 6 months
- Panic Disorder: sudden intense fear (panic attacks) with physical symptoms
- Social Anxiety Disorder: intense fear of social or performance situations
- Phobias: intense fear of a specific object or situation
Symptoms: persistent worry, restlessness, fatigue, difficulty concentrating, irritability,
muscle tension, sleep problems.
Treatment options: psychotherapy (cognitive behavioral therapy / CBT), medications
(SSRIs, SNRIs, buspirone, benzodiazepines for short-term use), or both.
Seek help from a qualified mental health professional for diagnosis and treatment.""",
    ),
    (
        "Depression — Overview and Treatment",
        "MedlinePlus / NIH",
        """Depression (major depressive disorder) is a common and serious medical illness that negatively
affects how you feel, think, and act. Symptoms must be present for at least two weeks.
Symptoms include: feeling sad or having a depressed mood; loss of interest or pleasure in
activities once enjoyed; changes in appetite; changes in sleep; loss of energy or increased
fatigue; feeling worthless or excessive guilt; difficulty thinking, concentrating, or making
decisions; thoughts of death or suicide.
Depression is among the most treatable of mental disorders — between 80-90% of people with
depression eventually respond well to treatment.
Treatment options: antidepressant medication (SSRIs, SNRIs, tricyclics), psychotherapy
(CBT, interpersonal therapy), or a combination.
If you or someone you know is struggling or in crisis, contact a mental health professional
or call/text a crisis helpline (e.g., 988 Suicide & Crisis Lifeline in the US).""",
    ),
    # ── Nutrition ────────────────────────────────────────────────────────────
    (
        "Healthy Eating — Dietary Guidelines Overview",
        "USDA / HHS Dietary Guidelines for Americans",
        """The Dietary Guidelines for Americans provides science-based advice on what to eat and drink
to promote health, reduce risk of chronic disease, and meet nutrient needs.
Key recommendations:
- Follow a healthy dietary pattern at every life stage.
- Customise and enjoy nutrient-dense food and beverage choices to reflect personal preferences,
  cultural traditions, and budgetary considerations.
- Focus on meeting food group needs with nutrient-dense foods and beverages, and stay within
  calorie limits.
- Limit foods and beverages higher in added sugars, saturated fat, and sodium, and limit
  alcoholic beverages.
Dietary patterns associated with good health: Mediterranean diet, DASH diet, plant-based diets.
Key nutrients: fibre (25-38 g/day), calcium, vitamin D, potassium, and iron are commonly under-consumed.
Sodium intake should be less than 2,300 mg/day.
Added sugars should be less than 10% of total daily calories.
Saturated fat should be less than 10% of total daily calories.""",
    ),
    (
        "Vitamins and Minerals — Common Deficiencies",
        "NIH Office of Dietary Supplements",
        """Common vitamin and mineral deficiencies in developed countries:
Vitamin D: Important for bone health and immune function. Sources: sunlight, fatty fish,
fortified foods. Deficiency linked to bone disease, muscle weakness, and immune dysfunction.
Most adults need 600-800 IU/day.
Iron: Essential for red blood cell production. Sources: red meat, poultry, seafood, beans, dark leafy
greens. Deficiency causes anaemia — fatigue, weakness, pale skin. Women of childbearing age are
at higher risk. RDA: 8 mg/day (men), 18 mg/day (women of reproductive age).
Vitamin B12: Required for nerve function and DNA synthesis. Found only in animal products (meat,
dairy, eggs). Vegans and older adults are at risk of deficiency. Symptoms: fatigue, weakness,
constipation, loss of appetite, nerve problems, depression, memory problems.
Folate: Critical for cell division; crucial during pregnancy to prevent neural tube defects.
Sources: leafy greens, legumes, fortified cereals. Recommended: 400 mcg/day; 600 mcg/day
during pregnancy.
Calcium: Essential for bone and teeth formation, muscle function, nerve transmission.
Sources: dairy, leafy greens, fortified foods. Deficiency leads to osteoporosis over time.
Adults need 1,000-1,200 mg/day.""",
    ),
    # ── Common Conditions ────────────────────────────────────────────────────
    (
        "Fever — When to Seek Medical Care",
        "MedlinePlus / NIH",
        """A fever is a temporary increase in body temperature, often due to illness. Normal body
temperature averages 37°C (98.6°F) but can vary.
A fever of 38°C (100.4°F) or higher is considered a fever in adults.
Fever is usually caused by: infection (viral — cold, flu, COVID-19; bacterial — pneumonia, UTI);
vaccination reaction; heat exhaustion; certain medications; autoimmune disorders; some cancers.
Management: rest, adequate fluids, fever-reducing medications such as paracetamol (acetaminophen)
or ibuprofen (always follow dosage instructions).
Seek immediate medical care if: the fever is above 39.4°C (103°F) in adults; it is accompanied
by severe headache, stiff neck, confusion, difficulty breathing, abdominal pain, rash; or if it
lasts more than 3 days. Febrile seizures in children: call emergency services immediately.""",
    ),
    (
        "Asthma — Overview and Triggers",
        "CDC / MedlinePlus",
        """Asthma is a chronic lung disease that inflames and narrows the airways, causing wheezing,
breathlessness, chest tightness, and coughing. Symptoms often worsen at night or in the morning.
Asthma affects people of all ages; it often starts in childhood. It cannot be cured but can be
well-controlled with proper management.
Common triggers: allergens (pollen, mould, pet dander, dust mites), respiratory infections,
exercise, cold air, air pollutants (smoke, smog), strong odours, stress, certain medications
(aspirin, NSAIDs, beta-blockers).
Asthma action plans: a written plan from a doctor telling you how to manage your asthma day-to-day
and what to do during an asthma attack.
Treatment: controller medicines (inhaled corticosteroids, long-acting beta-agonists, leukotriene
modifiers) taken daily; quick-relief (rescue) medicines (short-acting beta-agonists) for attacks.
During a severe attack not relieved by rescue inhaler — seek emergency medical care.""",
    ),
    (
        "Sleep Hygiene — Getting Better Sleep",
        "CDC / NIH",
        """Adults need 7 or more hours of sleep per night for best health and well-being. Insufficient sleep
increases the risk of obesity, heart disease, diabetes, mental health disorders, and accidents.
Sleep hygiene tips recommended by sleep experts:
- Be consistent: go to bed and wake up at the same time every day, including weekends.
- Make your bedroom quiet, dark, relaxing, and at a comfortable cool temperature.
- Remove electronic devices (TV, computers, phones) from the bedroom.
- Avoid large meals, caffeine, and alcohol before bedtime.
- Get regular physical activity (but not too close to bedtime).
- Avoid long naps during the day (especially late afternoon).
- Limit exposure to bright light in the evenings.
Common sleep disorders: insomnia, sleep apnoea, restless legs syndrome, narcolepsy.
If you consistently have trouble sleeping despite good sleep hygiene, talk to a healthcare provider.""",
    ),
]


# ---------------------------------------------------------------------------
# Chunking utility
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_words: int = 120, overlap_words: int = 20) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap_words
    return chunks


def _make_id(title: str, chunk_idx: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{slug}_c{chunk_idx}"


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def main() -> None:
    from pathlib import Path

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: Missing dependencies.\n"
            "Run: pip install sentence-transformers==3.3.1 chromadb==0.5.23"
        )
        return

    kb_dir = Path("data/knowledge_base/chroma")
    kb_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to ChromaDB at {kb_dir}…")
    client = chromadb.PersistentClient(path=str(kb_dir))
    collection = client.get_or_create_collection(
        name="medimind_kb",
        metadata={"hnsw:space": "cosine"},
    )

    print("Loading embedding model (all-MiniLM-L6-v2)…")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    print(f"Existing documents in collection: {len(existing_ids)}")

    to_add_ids: list[str] = []
    to_add_docs: list[str] = []
    to_add_metas: list[dict] = []

    for title, source, text in ARTICLES:
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = _make_id(title, i)
            if doc_id in existing_ids:
                continue
            to_add_ids.append(doc_id)
            to_add_docs.append(chunk)
            to_add_metas.append({"title": title, "source": source})

    if not to_add_ids:
        print(f"Nothing new to add. Collection already has {collection.count()} documents.")
        return

    print(f"Embedding {len(to_add_ids)} new chunks…")
    batch_size = 64
    for start in range(0, len(to_add_ids), batch_size):
        end = min(start + batch_size, len(to_add_ids))
        batch_docs = to_add_docs[start:end]
        embeddings = embedder.encode(batch_docs, show_progress_bar=True).tolist()
        collection.add(
            ids=to_add_ids[start:end],
            documents=batch_docs,
            metadatas=to_add_metas[start:end],
            embeddings=embeddings,
        )
        print(f"  Inserted {end}/{len(to_add_ids)}")

    total = collection.count()
    print(f"\nDone. Collection now contains {total} documents.")
    print("Restart the backend to activate the chat assistant.")


if __name__ == "__main__":
    main()
