import re

def remove_number_tokens(text):
    # Use regular expression to remove tokens made up of only numbers
    pattern = r'\b[0-9]+\b'  # Matches one or more digits (numbers) only
    return re.sub(pattern, '', text)

# Example usage:
input_text = "Hello 123 World! This is a test. 45678"
output_text = remove_number_tokens(input_text)
print(output_text)  # Output: "Hello  World! This is a test. "
