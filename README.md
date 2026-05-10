# Design_Agents_app

## Project Overview
The **Intelligent Multi-Agent Adaptive Learning System** is an advanced AI-driven tutoring platform designed to deliver hyper-personalized educational experiences. By leveraging a network of cooperating intelligent agents, the system dynamically tailors content delivery based on real-time student performance metrics.

### Key Features
* **Knowledge Modeling:** Utilizes probabilistic reasoning to analyze student interactions—including response latency, hint utilization, and historical accuracy—to map proficiency levels.
* **Adaptive Pedagogical Engine:** Employs **Reinforcement Learning** to optimize learning paths, adjust material difficulty, and select the most effective next-step interventions.
* **Automated Assessment:** Generates a diverse range of evaluations, including MCQ, True/False, Short Answer, and Essay questions.
* **Resource Recommendation:** Curates educational videos and documentation targeted at bridging identified conceptual gaps.
* **Analytics Dashboards:** Provides comprehensive progress tracking, concept prerequisite tracking, and automatic feedback for students.

---

## Prerequisites
Before run you must install the following dependencies:

```bash
pip install markitdown[all]
pip install python-dotenv
pip install spacy
---

## Development Workflow

### Styling & CSS Architecture
This project utilizes **Tailwind CSS**. The `style.css` file is a compiled output and is **automatically generated**. 

> **IMPORTANT:** Do not edit `style.css` directly. Any manual changes will be overwritten during the build process.

To modify the application's styling, follow these steps:

#### 1. Launch the CSS Compiler
Open your terminal in the project root directory and run the watcher script. This monitors your files and recompiles the CSS in real-time as you work:
```bash
npm run watch:css
```

#### 2. Edit the Input File
Apply all styling updates within `input.css`. This is the source file where you can add:
* Standard CSS rules.
* Tailwind `@apply` directives.
* Custom theme variables.

#### 3. Save and Synchronize
Upon saving `input.css`, the compiler will automatically detect the changes, process the Tailwind logic, and update the final `style.css` file for you.