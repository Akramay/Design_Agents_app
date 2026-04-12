const sections = document.querySelectorAll(".section");

function revealSections() {
  const triggerBottom = window.innerHeight * 0.75;

  sections.forEach(section => {
    const boxTop = section.getBoundingClientRect().top;

    if (boxTop < triggerBottom) {
      section.classList.add("active");
    }
  });
}

window.addEventListener("scroll", revealSections);

// trigger on load
revealSections();