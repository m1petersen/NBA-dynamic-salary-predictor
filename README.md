# NBA Valuation Intelligence

An interactive, data-driven predictive analytics platform designed to evaluate and project fair-market NBA player salaries. By analyzing comprehensive player production metrics against real-world contract structures, this platform surfaces market anomalies, identifies hyper-efficient assets, and isolates overvalued contracts.

## 📊 Core Features

- **Interactive Valuation Engine:** Visualizes actual contract salaries vs. model-predicted values using an interactive D3.js scatter plot with dynamic filtering across seasons.
- **Feature Importance Tracking:** Breaks down the specific algorithmic weights determining market value (e.g., Fantasy Points, Minutes Played, Age Curve).
- **Market Anomaly Spotting:** Dynamically isolates and ranks the league's top 5 most underpaid (Max Output Deficit) and overpaid (Max Output Surplus) players.
- **Player Sandbox Laboratory:** A real-time simulator allowing users to manipulate performance vectors (Age, PTS, AST, REB, Efficiency) to generate custom algorithmic market valuations instantly.

## 🤖 The Model Architecture
The underlying engine utilizes an ensemble machine learning approach—including **Random Forest** components to map complex career arcs, non-linear age curves, and production rates—trained on cross-validated multi-season NBA statistics. 

## 👥 Authors & Collaborators
Developed as a collaborative quantitative analytics project by:
- **Michael Petersen**
- **Rajveer Barring**
- **Nikhil Ganpule**

---
*Powered by Eclipse Analytics.*
