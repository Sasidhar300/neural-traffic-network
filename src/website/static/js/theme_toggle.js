// This script toggles between light and dark themes for a website.
// It saves the user's preference in localStorage and applies it on page load.

document.addEventListener('DOMContentLoaded', function () {
    const themeToggle = document.getElementById("light-mode");
    console.log(themeToggle);

    // Apply the saved theme on page load
    const savedTheme = localStorage.getItem("theme") || "dark"; // Default to dark mode
    document.documentElement.setAttribute("data-bs-theme", savedTheme);
    if (themeToggle) {
        themeToggle.checked = savedTheme === "light"; // Sync toggle state with saved theme
    }

    // Add event listener for theme toggle
    if (themeToggle) {
        themeToggle.addEventListener("change", () => { // Use "change" event for checkbox
            const newTheme = themeToggle.checked ? "light" : "dark";
            document.documentElement.setAttribute("data-bs-theme", newTheme);
            localStorage.setItem("theme", newTheme); // Save the new theme in localStorage
        });
    }

    // Initialize Bootstrap tooltips (Keep this here if you use tooltips)
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
