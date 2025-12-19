"""
Test script to verify dashboard explainer improvements
"""

from dashboard_explainer import get_dashboard_explainer

# Create a sample chart configuration
sample_chart_config = {
    "charts": [
        {
            "query": "Show ExtendedAmount by SalesTerritoryKey",
            "timestamp": "2025-01-04 10:30:00"
        }
    ],
    "title": "Sales Dashboard"
}

# Get explainer instance
explainer = get_dashboard_explainer()

# Test the chart description function
print("=" * 80)
print("Testing Chart Description Generation")
print("=" * 80)

query = "Show ExtendedAmount by SalesTerritoryKey"
description = explainer._describe_chart_from_query(query)
print(f"\nQuery: {query}")
print(f"\nGenerated Description:\n{description}")

# Test chart type inference
print("\n" + "=" * 80)
print("Testing Chart Type Inference")
print("=" * 80)

chart = {"query": query}
chart_type = explainer._infer_chart_type(chart)
print(f"\nInferred Chart Type: {chart_type}")

# Test question generation
print("\n" + "=" * 80)
print("Testing Question Generation")
print("=" * 80)

questions = explainer._generate_chart_questions([chart])
print(f"\nGenerated Question:\n{questions[0]}")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
