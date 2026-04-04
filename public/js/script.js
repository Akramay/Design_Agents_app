document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    reveal(); // Initial check for elements in view
});

// Scroll Reveal Animation Logic
function reveal() {
    var reveals = document.querySelectorAll(".reveal");
    for (var i = 0; i < reveals.length; i++) {
        var windowHeight = window.innerHeight;
        var elementTop = reveals[i].getBoundingClientRect().top;
        var elementVisible = 100; // threshold
        
        if (elementTop < windowHeight - elementVisible) {
            reveals[i].classList.add("active");
        }
    }
}

// Trigger reveal on scroll
window.addEventListener("scroll", reveal);

// Parallax Effect for Hero Visual
document.addEventListener('mousemove', (e) => {
    const visual = document.getElementById('hero-visual');
    if(visual && window.innerWidth > 768) {
        const x = (e.clientX / window.innerWidth - 0.5) * 20; // 20px max movement
        const y = (e.clientY / window.innerHeight - 0.5) * 20;
        
        // Select floating cards to apply slight inverse parallax
        const cards = visual.querySelectorAll('.glass');
        cards.forEach((card, index) => {
            const factor = index % 2 === 0 ? 1 : -1;
            card.style.transform = `translate(${x * factor}px, ${y * factor}px)`;
        });
    }
});