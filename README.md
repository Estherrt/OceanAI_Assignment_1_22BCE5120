# OceanAI_Assignment_1
Development of an Autonomous QA Agent for Test Case and Script Generation

Description:
An AI-powered system to automate software test case generation and Selenium script creation from support documents and HTML content.

Setup Instructions:
1. Clone the repository:
git clone https://github.com/Estherrt/OceanAI_Assignment_1_22BCE5120.git
cd <repository-folder>

2. Install dependencies from the provided requirements.txt:
pip install -r requirements.txt

Running the application:
streamlit run app.py

Usage Examples:

Phase 1 – Knowledge Base Ingestion
Upload support documents (.md, .txt, .pdf, .json) or paste HTML content.
Configure chunk size and overlap for text splitting.
Click Build Knowledge Base to store document embeddings in ChromaDB.

Phase 2 – Generate Test Cases
Ask the QA Agent a question like:
"Generate positive and negative test cases for discount code feature"
The system retrieves relevant knowledge chunks and generates structured JSON or Markdown table test cases.

Phase 3 – Selenium Script Generation
Select a previously generated test case.
Click Generate Selenium Script.
Python Selenium script is generated in a code block.

Explanation of support documents:
product_spec.md – Product specifications and rules
Product List: Includes Black Tank Top, Beige Cargo Pants, Black & White Striped Shoes with prices and stock status
Cart Rules: Add, remove, and update quantity
Pricing Rules: Total calculation and discount application
Discount Codes: DISCOUNTFOR16 (16%) and HALFPRICEOFF (50%)
Shipping Rules: Standard (free) and Express (+Rs.100)
User Details: Name, Email, Address validation
Payment: Credit Card and PayPal, with success/failure messages

ui_ux_guide.txt – UI/UX design guidelines
Page Layout: Light blue background, Times New Roman font, centered title
Product Section: Flex layout, product-card styling, Add to Cart button design
Cart Section: Dynamic updates, item layout, empty cart message
Price, Discount, Shipping: Auto-updates and visual feedback
User Details Form: Label above each field, instant validation, error messages in red
Payment Section: Radio buttons for options, color styling, success/failure messages


Esther Rachel Thomas
November 2025
