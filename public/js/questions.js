const courses = {
    cs: ["Data Structures", "AI", "Algorithms"],
    eng: ["Mechanics", "Thermodynamics"],
    business: ["Marketing", "Finance"]
};

const LETTERS = ["A", "B", "C", "D"];

/* ================= QUIZ DATA ================= */
const quizData = [
    {
        q: "What is the time complexity of binary search?",
        options: ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
        correct: 1,
        hint: "Array is divided in half each step."
    },
    {
        q: "Which data structure uses FIFO?",
        options: ["Stack", "Queue", "Tree", "Graph"],
        correct: 1,
        hint: "First In First Out."
    },
    {
        q: "Which algorithm finds the shortest path in weighted graphs?",
        options: ["DFS", "BFS", "Dijkstra", "Bubble Sort"],
        correct: 2,
        hint: "Used for weighted graphs."
    }
];

let currentQuestion = 0;
let score = 0;

/* ================= STEP NAVIGATION ================= */
function goToCourses() {
    const dept = document.getElementById("department").value;
    if (!dept) return alert("Please select a department first.");

    const courseSelect = document.getElementById("course");
    courseSelect.innerHTML = "";

    courses[dept].forEach(c => {
        const option = document.createElement("option");
        option.value = c;
        option.textContent = c;
        courseSelect.appendChild(option);
    });

    document.getElementById("step1").classList.add("hidden");
    document.getElementById("step2").classList.remove("hidden");
}

function goBack(step) {
    document.getElementById("step" + (step + 1)).classList.add("hidden");
    document.getElementById("step" + step).classList.remove("hidden");
}

/* ================= START QUIZ ================= */
function startQuestions() {
    const course = document.getElementById("course").value;

    currentQuestion = 0;
    score = 0;

    document.getElementById("topicBadge").textContent = course;

    document.getElementById("step2").classList.add("hidden");
    document.getElementById("step3").classList.remove("hidden");

    loadQuestion();
}

/* ================= LOAD QUESTION ================= */
function loadQuestion() {
    const q = quizData[currentQuestion];

    document.getElementById("questionText").textContent = q.q;
    document.getElementById("hintText").textContent = q.hint;

    document.getElementById("questionCounter").textContent =
        `Question ${currentQuestion + 1} of ${quizData.length}`;

    document.getElementById("progressFill").style.width =
        ((currentQuestion + 1) / quizData.length) * 100 + "%";

    document.getElementById("hintBox").classList.add("hidden");

    const nextBtn = document.getElementById("nextBtn");
    if (nextBtn) nextBtn.classList.add("hidden");

    const optionsDiv = document.getElementById("options");
    optionsDiv.innerHTML = "";

    q.options.forEach((opt, i) => {
        const div = document.createElement("div");
        div.className = "option";
        div.innerHTML = `
            <span class="option-letter">${LETTERS[i]}</span>
            <span class="option-text">${opt}</span>
        `;

        div.onclick = () => handleAnswer(div, i);
        optionsDiv.appendChild(div);
    });
}

/* ================= HANDLE ANSWER ================= */
function handleAnswer(selectedEl, index) {
    const correctIndex = quizData[currentQuestion].correct;

    document.querySelectorAll(".option").forEach(el => {
        el.style.pointerEvents = "none";
    });

    if (index === correctIndex) {
        selectedEl.classList.add("correct");
        score++;
    } else {
        selectedEl.classList.add("wrong");

        document.querySelectorAll(".option")[correctIndex].classList.add("correct");
    }

    const nextBtn = document.getElementById("nextBtn");
    if (nextBtn) nextBtn.classList.remove("hidden");
}

/* ================= NEXT QUESTION ================= */
function nextQuestion() {
    currentQuestion++;

    if (currentQuestion >= quizData.length) {
        showResult();
    } else {
        loadQuestion();
    }
}

/* ================= RESULT ================= */
function showResult() {
    document.getElementById("step3").classList.add("hidden");

    const resultScreen = document.getElementById("resultScreen");
    if (resultScreen) {
        resultScreen.classList.remove("hidden");

        document.getElementById("scoreText").textContent =
            `You scored ${score} out of ${quizData.length}`;
    } else {
        alert(`Quiz Finished! Score: ${score}/${quizData.length}`);
    }
}

/* ================= HINT ================= */
function toggleHint() {
    const hintBox = document.getElementById("hintBox");
    const btn = document.querySelector(".hint-btn");

    const isHidden = hintBox.classList.contains("hidden");

    hintBox.classList.toggle("hidden");
    btn.textContent = isHidden ? "💡 Hide Hint" : "💡 Show Hint";
}

/* ================= LUCIDE ICONS ================= */
if (typeof lucide !== "undefined") {
    lucide.createIcons();
}

/* ================= THEME ================= */
const themeBtn = document.getElementById("theme-toggle");
const iconSun  = document.getElementById("icon-sun");
const iconMoon = document.getElementById("icon-moon");

function applyTheme(isLight) {
    document.body.classList.toggle("light-mode", isLight);
    if (iconSun && iconMoon) {
        iconSun.style.display  = isLight ? "none" : "";
        iconMoon.style.display = isLight ? ""     : "none";
    }
}

// Load saved preference (default = dark)
const savedTheme = localStorage.getItem("theme");
applyTheme(savedTheme === "light");

if (themeBtn) {
    themeBtn.addEventListener("click", () => {
        const isLight = !document.body.classList.contains("light-mode");
        applyTheme(isLight);
        localStorage.setItem("theme", isLight ? "light" : "dark");
    });
}

/* ================= MOBILE MENU ================= */
const menuBtn    = document.getElementById("mobile-menu-btn");
const mobileMenu = document.getElementById("mobile-menu");
const iconMenu   = document.getElementById("icon-menu");
const iconClose  = document.getElementById("icon-close");

if (menuBtn && mobileMenu) {
    menuBtn.addEventListener("click", () => {
        const isOpen = mobileMenu.classList.toggle("open");
        iconMenu.style.display  = isOpen ? "none" : "";
        iconClose.style.display = isOpen ? ""     : "none";
    });
}