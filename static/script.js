// script.js

document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyze-btn");
  const newsText = document.getElementById("news-text");
  const resultContainer = document.getElementById("result-container");
  const resultBox = document.getElementById("result-box");
  const predictionLabel = document.getElementById("prediction-label");
  const confidenceScore = document.getElementById("confidence-score");

  // Get the new loader element
  const loader = document.getElementById("loader-container");

  analyzeBtn.addEventListener("click", () => {
    const text = newsText.value;

    if (text.trim() === "") {
      alert("Please paste some text to analyze.");
      return;
    }

    // Show loading state
    loader.classList.remove("hidden");
    resultContainer.classList.add("hidden"); // Hide previous results
    resultBox.className = ""; // Reset result box colors

    // Send data to the /predict endpoint
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: text }),
    })
      .then((response) => {
        if (!response.ok) {
          // Handle HTTP errors like 500
          throw new Error(`Server error: ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        resultContainer.classList.remove("hidden"); // Show result container

        if (data.error) {
          // Handle server-side application error
          predictionLabel.textContent = "Error";
          confidenceScore.textContent = data.error;
          resultBox.classList.add("result-fake");
        } else {
          // Display success
          predictionLabel.textContent = data.prediction;
          confidenceScore.textContent = `Confidence: ${data.confidence}`;

          // Apply styles based on prediction
          if (data.prediction === "Real") {
            resultBox.classList.add("result-real");
          } else {
            resultBox.classList.add("result-fake");
          }
        }
      })
      .catch((error) => {
        // Handle network error or fetch-related issues
        console.error("Error:", error);
        resultContainer.classList.remove("hidden"); // Show container to display error
        predictionLabel.textContent = "Error";
        confidenceScore.textContent = "Could not connect to the server.";
        resultBox.classList.add("result-fake");
      })
      .finally(() => {
        // This block runs after .then() or .catch()
        // Always hide the loader
        loader.classList.add("hidden");
      });
  });
});
