const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultCard = document.getElementById("resultCard");
const resultText = document.getElementById("resultText");
const loading = document.getElementById("loading");
const themeBtn = document.getElementById("theme-toggle");
const iconSun = document.getElementById("icon-sun");
const iconMoon = document.getElementById("icon-moon");

/* INITIALIZE ICONS */
lucide.createIcons();

/* FILE INPUT CHANGE */
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        analyzeBtn.disabled = false;
        document.querySelector('.upload-content span').innerText = fileInput.files[0].name;
    } else {
        analyzeBtn.disabled = true;
    }
});

/* FAKE AI LOGIC */
analyzeBtn.addEventListener("click", () => {
    resultCard.classList.remove("hidden");
    loading.classList.remove("hidden");
    resultText.innerHTML = "";
    
    analyzeBtn.innerText = "Analyzing...";
    analyzeBtn.disabled = true;

    setTimeout(() => {
        loading.classList.add("hidden");
        analyzeBtn.innerText = "Generate Explanation →";
        analyzeBtn.disabled = false;

        resultText.innerHTML = `
            <div style="text-align: left; max-width: 500px; margin: 0 auto; line-height: 1.6;">
                <p><strong>Summary:</strong> High-level breakdown of your content.</p>
                <ul style="padding-left: 20px;">
                    <li>Main concepts identified.</li>
                    <li>Technical jargon simplified.</li>
                    <li>Logical flow established.</li>
                </ul>
            </div>
        `;
    }, 1500);
});

/* SMOOTH THEME TOGGLE */
function updateThemeIcons(isLight) {
    iconSun.style.display = isLight ? "none" : "block";
    iconMoon.style.display = isLight ? "block" : "none";
}

themeBtn.addEventListener("click", () => {
    const isLight = document.body.classList.toggle("light-mode");
    localStorage.setItem("theme", isLight ? "light" : "dark");
    updateThemeIcons(isLight);
});

// Initialize icons based on current state
updateThemeIcons(document.body.classList.contains("light-mode"));