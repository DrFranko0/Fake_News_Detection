document.getElementById("newsForm").addEventListener("submit", async function (e) {
    e.preventDefault();
  
    const newsText = document.getElementById("newsText").value;
    const resultDiv = document.getElementById("result");
    
    resultDiv.classList.add("hidden");
    resultDiv.textContent = "Processing...";
  
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: newsText }),
    });
  
    if (response.ok) {
      const data = await response.json();
      resultDiv.textContent = `Result: ${data.result}`;
      resultDiv.classList.remove("hidden");
    } else {
      resultDiv.textContent = "An error occurred. Please try again.";
      resultDiv.classList.remove("hidden");
    }
  });
  