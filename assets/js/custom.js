document.addEventListener("DOMContentLoaded", function () {
  // Select the element. Adjust the selector as needed.
  // If you can add a custom class (e.g., "animated-text"), use that:
  // const element = document.querySelector('.animated-text');

  // Otherwise, select by the Wowchemy classes.
  // Note: In CSS selectors, colons need to be escaped with a backslash.
  const element = document.querySelector(".mb-6.text-3xl.font-bold.text-gray-900.dark\\:text-white");

  if (!element) return; // Exit if the element is not found

  // Get the text content of the element.
  const text = element.textContent;

  // Clear the existing content.
  element.innerHTML = "";

  // Loop through each character and wrap it in a span.
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === " ") {
      // Preserve spaces
      element.innerHTML += " ";
    } else {
      // Wrap non-space characters in a span.
      element.innerHTML += `<span>${char}</span>`;
    }
  }
});
