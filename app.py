import os
import json
import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

# ── Configure Groq ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

# ── Optional Supabase ─────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase_client = None

# ── Role → Expected Skills Map (for skill gap analysis) ──────────────────────
ROLE_SKILLS_MAP = {
    "data scientist": ["Python", "SQL", "Machine Learning", "Statistics", "Pandas", "NumPy",
                       "TensorFlow", "Data Visualization", "Feature Engineering", "Deep Learning"],
    "software engineer": ["Python", "JavaScript", "Data Structures", "Algorithms", "Git",
                          "REST APIs", "SQL", "System Design", "Testing", "Docker"],
    "web developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Git",
                      "REST APIs", "Responsive Design", "TypeScript", "SQL"],
    "machine learning engineer": ["Python", "TensorFlow", "PyTorch", "Scikit-learn",
                                   "MLOps", "Docker", "SQL", "Statistics", "Deep Learning", "Kubernetes"],
    "data analyst": ["SQL", "Python", "Excel", "Power BI", "Tableau", "Statistics",
                     "Data Cleaning", "Pandas", "Communication", "R"],
    "devops engineer": ["Docker", "Kubernetes", "CI/CD", "Linux", "AWS", "Git",
                        "Ansible", "Terraform", "Python", "Monitoring"],
    "full stack developer": ["HTML", "CSS", "JavaScript", "React", "Node.js",
                              "SQL", "REST APIs", "Git", "Docker", "TypeScript"],
    "android developer": ["Java", "Kotlin", "Android SDK", "REST APIs", "Git",
                           "Firebase", "XML", "Jetpack Compose", "SQL", "Testing"],
    "ios developer": ["Swift", "Xcode", "UIKit", "SwiftUI", "REST APIs",
                      "Git", "Core Data", "Firebase", "Testing", "Objective-C"],
    "cybersecurity analyst": ["Networking", "Linux", "Python", "Penetration Testing",
                              "SIEM", "Cryptography", "Firewalls", "Incident Response", "Compliance", "SQL"],
}

LEARNING_RESOURCES = {
    "Python": "https://www.learnpython.org",
    "SQL": "https://sqlbolt.com",
    "Machine Learning": "https://www.coursera.org/learn/machine-learning",
    "Statistics": "https://www.khanacademy.org/math/statistics-probability",
    "TensorFlow": "https://www.tensorflow.org/tutorials",
    "PyTorch": "https://pytorch.org/tutorials",
    "React": "https://react.dev/learn",
    "Docker": "https://docs.docker.com/get-started",
    "Kubernetes": "https://kubernetes.io/docs/tutorials",
    "Git": "https://learngitbranching.js.org",
    "JavaScript": "https://javascript.info",
    "TypeScript": "https://www.typescriptlang.org/docs",
    "Node.js": "https://nodejs.dev/learn",
    "AWS": "https://aws.amazon.com/training/free",
    "Linux": "https://linuxjourney.com",
    "Tableau": "https://www.tableau.com/learn/training",
    "Power BI": "https://learn.microsoft.com/en-us/power-bi",
    "Pandas": "https://pandas.pydata.org/docs/getting_started",
    "Deep Learning": "https://www.deeplearning.ai/courses",
    "System Design": "https://github.com/donnemartin/system-design-primer",
    "REST APIs": "https://restfulapi.net",
    "Swift": "https://developer.apple.com/swift/resources",
    "Kotlin": "https://kotlinlang.org/docs/home.html",
    "Penetration Testing": "https://www.hackthebox.com",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def calculate_ats_score(resume_text: str, role: str, skills: list) -> int:
    """Calculate a simple ATS score based on keyword presence."""
    text_lower = resume_text.lower()
    role_keywords = role.lower().split()
    role_hits = sum(1 for kw in role_keywords if kw in text_lower)
    role_score = min(role_hits / max(len(role_keywords), 1), 1) * 40
    skill_hits = sum(1 for s in skills if s.lower() in text_lower)
    skill_score = min(skill_hits / max(len(skills), 1), 1) * 60
    return round(role_score + skill_score)


def get_skill_gaps(role: str, user_skills: list) -> list:
    """Compare user skills against expected skills for the role."""
    role_key = role.lower().strip()
    expected = []
    for key, skills in ROLE_SKILLS_MAP.items():
        if key in role_key or role_key in key:
            expected = skills
            break
    if not expected:
        expected = ["Communication", "Problem Solving", "Git", "Documentation", "Testing"]
    user_skills_lower = [s.lower().strip() for s in user_skills]
    gaps = []
    for skill in expected:
        if skill.lower() not in user_skills_lower:
            gaps.append({
                "skill": skill,
                "resource": LEARNING_RESOURCES.get(
                    skill,
                    f"https://www.google.com/search?q=learn+{skill.replace(' ', '+')}"
                ),
            })
    return gaps[:6]


def generate_resume(
    name: str, role: str, skills: list,
    email: str = "", phone: str = "", location: str = "",
    college: str = "", degree: str = "", grad_year: str = "",
    linkedin: str = "", languages: str = "", projects: str = "",
    purpose: str = "", year_of_study: str = "",
    experience: str = "", template: str = "modern"
) -> str:
    """Call Groq (Llama 3.3) to generate a professional resume."""
    if not client:
        return "ERROR: Groq API key not configured. Add GROQ_API_KEY to your .env file."

    skills_str = ", ".join(skills)
    name_str = name or "Your Name"

    # Build contact line
    contact_parts = []
    if email:    contact_parts.append(f"Email: {email}")
    if phone:    contact_parts.append(f"Phone: {phone}")
    if location: contact_parts.append(f"Location: {location}")
    if linkedin: contact_parts.append(f"LinkedIn: {linkedin}")
    contact_str = " | ".join(contact_parts) if contact_parts else "(placeholder contact info)"

    # Education line
    edu_parts = []
    if college:   edu_parts.append(college)
    if degree:    edu_parts.append(degree)
    if grad_year: edu_parts.append(f"Graduating {grad_year}")
    edu_str = ", ".join(edu_parts) if edu_parts else "(placeholder education)"

    # Purpose-specific tone instructions
    purpose_map = {
        "Club Interview":  "This resume is for a college club/society interview. Highlight leadership, teamwork, extracurriculars, and relevant projects. Keep it concise and enthusiastic.",
        "Internship":      "This is for an internship application. Emphasise academic projects, technical skills, and eagerness to learn. Include relevant coursework if applicable.",
        "Job Application": "This is for a full-time job. Emphasise quantified achievements, professional experience, and impact metrics.",
        "Freelance":       "This is for freelance/gig work. Highlight portfolio projects, client-facing skills, and self-motivation.",
        "Other":           "Write a general-purpose professional resume.",
    }
    purpose_note = purpose_map.get(purpose, "Write a professional resume.")

    # Template style description
    template_style_map = {
        "modern":    "bold section headers, tech-forward language, modern formatting",
        "classic":   "traditional formal structure, conservative language",
        "minimal":   "ultra-clean with ample white space, brevity is key",
        "technical": "technical depth, include tech stacks, tools and frameworks prominently",
        "executive": "authoritative tone, leadership focus, quantified business impact metrics, serif-ready layout",
        "creative":  "energetic and expressive tone, strong action verbs, highlight creativity and innovation",
        "academic":  "formal scholarly tone, emphasise education, research, publications and academic achievements",
        "startup":   "bold and disruptive tone, highlight growth mindset, side projects, and entrepreneurial impact",
        "elegant":   "refined and sophisticated language, highlight soft skills alongside technical expertise",
    }
    template_style = template_style_map.get(template, "clean and professional")

    # Languages section line
    languages_section = f"8. LANGUAGES ({languages})" if languages else ""

    prompt = (
        "You are a world-class professional resume writer. Create a polished, ATS-optimized resume.\n\n"
        "== CANDIDATE INFORMATION ==\n"
        f"Name: {name_str}\n"
        f"Contact: {contact_str}\n"
        f"Education: {edu_str}\n"
        f"Year of Study: {year_of_study or 'Not specified'}\n"
        f"Experience Level: {experience or 'Not specified'}\n"
        f"Languages: {languages or 'Not specified'}\n"
        f"Target Role: {role}\n"
        f"Skills: {skills_str}\n"
        f"Additional Projects/Achievements: {projects or 'None provided'}\n\n"
        "== RESUME PURPOSE ==\n"
        f"{purpose_note}\n\n"
        "== TEMPLATE STYLE ==\n"
        f"Format: {template.title()} - {template_style}\n\n"
        "== SECTIONS TO INCLUDE ==\n"
        "1. CONTACT INFORMATION (use the real contact data provided above)\n"
        f"2. PROFESSIONAL SUMMARY (3 punchy sentences, tailored to {role} and {purpose or 'the role'})\n"
        "3. TECHNICAL SKILLS (grouped by category, e.g., Languages / Frameworks / Tools)\n"
        f"4. WORK EXPERIENCE (2-3 positions relevant to {role}; bullet points with action verbs and metrics)\n"
        "5. PROJECTS (2 relevant projects - if candidate mentioned any above, use them; otherwise generate realistic ones with tech stack)\n"
        "6. EDUCATION (use the real education info provided)\n"
        f"7. CERTIFICATIONS (2-3 relevant certifications for {role})\n"
        f"{languages_section}\n\n"
        "== RULES ==\n"
        "- Use REAL contact/education data provided, not placeholders\n"
        "- Use strong action verbs: Developed, Engineered, Implemented, Optimized, Led, Built\n"
        "- Add realistic metrics (improved performance by 30%, reduced deployment time by 50%)\n"
        "- Keep to 1 page of content total\n"
        f"- Make it ATS-friendly with keywords for: {role}\n"
        "- Section headers in ALL CAPS\n"
        "- Output ONLY the resume text - no commentary, no markdown, no explanations."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return "ERROR: Groq API rate limit hit. Please wait a moment and try again."
        raise


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received."}), 400

        # Required fields
        role       = data.get("role", "").strip()
        skills_raw = data.get("skills", "").strip()
        if not role:       return jsonify({"error": "Role is required."}), 400
        if not skills_raw: return jsonify({"error": "Skills are required."}), 400

        # All form fields
        name          = data.get("name", "").strip()
        email         = data.get("email", "").strip()
        phone         = data.get("phone", "").strip()
        location      = data.get("location", "").strip()
        college       = data.get("college", "").strip()
        degree        = data.get("degree", "").strip()
        grad_year     = data.get("grad_year", "").strip()
        linkedin      = data.get("linkedin", "").strip()
        languages     = data.get("languages", "").strip()
        projects      = data.get("projects", "").strip()
        year_of_study = data.get("year_of_study", "").strip()
        purpose       = data.get("purpose", "").strip()
        experience    = data.get("experience", "").strip()
        template      = data.get("template", "modern").strip()

        skills = [s.strip() for s in re.split(r"[,;]", skills_raw) if s.strip()]

        # Generate resume via Groq
        resume_text = generate_resume(
            name=name, role=role, skills=skills,
            email=email, phone=phone, location=location,
            college=college, degree=degree, grad_year=grad_year,
            linkedin=linkedin, languages=languages, projects=projects,
            purpose=purpose, year_of_study=year_of_study,
            experience=experience, template=template
        )
        if resume_text.startswith("ERROR:"):
            return jsonify({"error": resume_text}), 500

        ats_score  = calculate_ats_score(resume_text, role, skills)
        skill_gaps = get_skill_gaps(role, skills)

        result = {"resume": resume_text, "ats_score": ats_score, "skill_gaps": skill_gaps}

        # Save to Supabase (optional)
        if supabase_client:
            try:
                supabase_client.table("resumes").insert({
                    "name": name or "Anonymous", "role": role,
                    "skills": skills_raw, "ats_score": ats_score,
                    "resume_text": resume_text,
                }).execute()
            except Exception:
                pass

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\nAI Resume Builder is running!")
    print(f"Open your browser -> http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
