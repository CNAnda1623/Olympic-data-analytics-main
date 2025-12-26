# 🏅 Olympic Data Analytics

A Python-based data analytics project that explores and visualizes **historical Olympic Games data** to uncover trends, patterns, and insights related to athletes, countries, events, and medal distributions.

This project focuses on **data analysis, visualization, and storytelling**, demonstrating how raw CSV datasets can be transformed into meaningful insights using Python.

---

## 🌟 Project Overview

The Olympic Data Analytics project analyzes historical Olympic datasets containing athlete participation, events, and country (NOC) information. The system processes these datasets to generate analytical views that help answer questions such as:

* How countries have performed over time
* Participation trends across different Olympic editions
* Athlete-level and event-level insights

The project is designed for **learning and analytical practice**, not as a production-grade analytics platform.

---

## ✨ Key Features

### 📊 Data Analysis

* Analysis of athlete participation across Olympic Games
* Country-wise (NOC) performance insights
* Event and sport-level breakdowns

### 📈 Visual Insights

* Graphical representations of trends and distributions
* Comparative analysis between countries and years
* Clean visual outputs for easier interpretation

### 🗂️ Structured Dataset Handling

* Uses CSV-based Olympic datasets
* Clean separation of raw data and processed logic
* Reusable analysis functions

### 🧩 Modular Python Design

* Separate scripts for landing logic and analysis views
* Easy-to-read and extendable code structure

---

## 🛠️ Tech Stack

### Core Technologies

* **Python**

### Libraries & Tools

* **Pandas** – data manipulation
* **Matplotlib / Seaborn** – data visualization
* **NumPy** – numerical operations

---

## 🏗️ Project Structure

```
Olympic-data-analytics-main/
├── archive/                   # Original Olympic datasets
│   ├── athlete_events.csv
│   └── noc_regions.csv
│
├── csv_files/                  # Working CSV copies
│   ├── athlete_events.csv
│   └── noc_regions.csv
│
├── images/                     # Project images
│   └── olympic_image.jpg
│
├── landing_page.py             # Entry / overview logic
├── analysis_view.py            # Data analysis & visualization
├── main.py                     # Main execution script
└── README.md
```

---

## 🔁 Application Flow (Simplified)

1. Olympic CSV datasets are loaded
2. Data is cleaned and prepared for analysis
3. Analytical computations are performed
4. Visualizations are generated
5. Insights are displayed or saved

---

## ▶️ Running the Project Locally

### Prerequisites

* Python 3.8+

### Setup & Run

```bash
pip install pandas matplotlib seaborn numpy
python main.py
```

---

## ⚠️ Known Limitations

* Static datasets (no live updates)
* Limited to historical Olympic data provided
* Visualizations are exploratory, not interactive dashboards

---

## 🧪 Troubleshooting (Optional)

* **CSV file not found**

  * Ensure file paths match directory structure

* **Plots not displaying**

  * Run the script in an environment that supports graphical output

---

## 🎯 Learning Outcomes

* Working with real-world sports datasets
* Applying data cleaning and preprocessing techniques
* Visualizing large datasets effectively
* Drawing insights from historical data

---

**Olympic Data Analytics – Turning sports data into meaningful insights**
