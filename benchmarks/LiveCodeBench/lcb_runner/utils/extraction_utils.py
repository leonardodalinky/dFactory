import re

from lcb_runner.lm_styles import LMStyle
from typing import List, Dict, Optional
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")

def parse_interleaved_response(text: str) -> List[Dict[str, str]]:
    """Parse a raw interleaved assistant response into a list of icontent dicts.

    Handles common malformations from model generation:
    - Missing ``<ithink>`` or ``<ioutput>`` opening tags
    - Missing ``<iconv>`` wrapper tags
    - Output-only blocks (no think content)

    Parameters
    ----------
    text : str
        Raw decoded text of the assistant response (may include special tokens
        such as ``<|role_end|>``, ``<|endoftext|>``, ``<|interleaved|>``).

    Returns
    -------
    list[dict]
        Each element is ``{"think": str, "output": str}``.
    """
    # Strip special tokens and role markers
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = re.sub(r"<role>.*?</role>", "", text)
    text = text.strip()

    # Split into blocks by <iconv> / </iconv> boundaries
    blocks = re.split(r"</?iconv>", text)
    blocks = [b for b in blocks if b.strip()]

    results: List[Dict[str, str]] = []
    for block in blocks:
        think = ""
        output = ""

        # Extract think content (try proper tag first, fall back to missing opening tag)
        think_match = re.search(r"<ithink>(.*?)</ithink>", block, re.DOTALL)
        if not think_match:
            think_match = re.search(r"(.*?)</ithink>", block, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
            remaining = block[think_match.end():]
        else:
            remaining = block

        # Extract output content (try proper tag first, fall back to missing opening tag)
        output_match = re.search(r"<ioutput>(.*?)</ioutput>", remaining, re.DOTALL)
        if not output_match:
            output_match = re.search(r"(.*?)</ioutput>", remaining, re.DOTALL)
        if output_match:
            output = output_match.group(1).strip()
        elif remaining.strip():
            # No output tags at all — treat the remainder as output
            output = remaining.strip()

        # Clean any stray opening tags left over from malformed generation
        think = re.sub(r"^<ithink>", "", think).strip()
        output = re.sub(r"^<ioutput>", "", output).strip()

        if think or output:
            results.append({"think": think, "output": output})

    return results

def extract_code(model_output: str, lmstyle: LMStyle):
    if lmstyle == LMStyle.iCoder:
        # from models.llada2_moe.parse_response import parse_interleaved_response

        blocks = parse_interleaved_response(model_output)
        parts = []
        for block in blocks:
            text = block["output"]
            if not text:
                continue
            # Strip markdown code fences the model may wrap around output blocks
            text = re.sub(r"^```[\w]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            parts.append(text)
        return "\n".join(parts)
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    elif lmstyle == LMStyle.GenericBase:
        return model_output.strip()
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return ""
        # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def extract_test_output_code(model_output: str, lmstyle: LMStyle = None):
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    if lmstyle and lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        # first try to extract ```python if not then try ```
        indexlines = [
            i
            for i, line in enumerate(outputlines)
            if "```python" in line or "```Python" in line
        ]
        if indexlines:
            start_index = indexlines[0]
        else:
            start_index = None
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if start_index is not None:
            indexlines = [i for i in indexlines if i > start_index]
            indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def extract_execution_code(model_output: str, lmstyle: LMStyle, cot: bool = False):
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()

if __name__ == "__main__":
    # print(extract_code("\n<iconv>\n<ithink>The problem statement requires a function that computes the maximum possible score when jumping through an array, and the full program defines a class with that method. The first code segment should introduce the module and import the necessary type hint for the function's parameter. Adding the import for List from typing is essential because the method signature uses List[int] as the parameter annotation, and this import must appear before the class definition to satisfy the type hint. Therefore, this segment establishes the module header and brings in the typing import, setting up the context for the subsequent class implementation.</ithink>\n<ioutput>```python\n# findMaximumScore.py\nfrom typing import List</ioutput>\n</iconv>\n<iconv>\n<ithink>The problem asks for a function that computes the maximum possible score when jumping through an array of integers. The full program defines a class with a method named findMaximumScore that takes a list of integers and returns an integer. This code segment is the beginning of that method, establishing the function signature and docstring. It is necessary to introduce the method, document its parameters and return type, and provide a brief description of its purpose, so that subsequent chunks can implement the logic while keeping the code organized and self\u2011documenting.</ithink>\n<ioutput>class Solution:\n    def findMaximumScore(self, nums: List[int]) -> int:\n        \"\"\"\n        Find the maximum score by jumping through the array.</ioutput>\n</iconv>\n<iconv>\n<ithink>The function signature and docstring have just been introduced, so the next logical step is to document the parameters and return value. This segment provides the type annotations for the `nums` argument and the return type, matching the method's purpose. It also includes a concise description of the `nums` parameter and the return value, ensuring that the generated code is self\u2011explanatory and aligns with the overall design of the `findMaximumScore` method.</ithink>\n<ioutput>:param nums: List[int] - The array of integers.\n        :return: int - The maximum score.\n        \"\"\"</ioutput>\n</iconv>\n<iconv>\n<ithink>The function needs to know the size of the input array to correctly bound the loop that processes each element. By assigning the length of the nums list to a variable (n) early, the subsequent code can reference this value for the while\u2011loop condition (index < n) and for any future calculations such as tracking the maximum score. This step is essential before the iteration begins, ensuring the algorithm can safely traverse the entire array without out\u2011of\u2011bounds errors.</ithink>\n<ioutput>n = len(nums)</ioutput>\n</iconv>\n<iconv>\n<ithink>The function findMaximumScore must track the best result seen so far while iterating through the array. Before the loop starts, we need a variable that will hold the highest score encountered. Initializing max_score to 0 provides a neutral starting point, that that any positive values in the array will be captured and that the loop can safely compare each new score against this baseline. This initialization is essential for the subsequent while\u2011loop to correctly update the maximum score.</ithink>\n<ioutput>max_score = 0</ioutput>\n</iconv>\n<iconv>\n<ithink>The function findMaximumScore must iterate over the input list to evaluate each element as a potential score. This segment introduces the while loop that drives the traversal, using an index variable initialized to 0. The condition `index < n` ensures the loop runs for every element in the nums array, preventing out\u2011of\u2011bounds access. Inside the loop, `current_score = nums[index]` extracts the current element's value, which is essential for the subsequent comparison that updates the maximum score. Generating this code maintains the logical flow of the algorithm and aligns with the surrounding code that initializes `index` and `max_score`.</ithink>\n<ioutput>while index < n:\n        current_score = nums[index]</ioutput>\n</iconv>\n<iconv>\n<ithink>In the findMaximumScore function we need to track the highest value seen so far while iterating through the input list. The line that updates max_score is essential because it implements the core comparison logic: if the current element's score exceeds the previously recorded maximum, we replace max_score with the new value. This ensures the loop correctly computes the overall maximum score across the entire array, which is the function's purpose. Generating this line maintains the correctness of the algorithm's state\u2011updating step within the while\u2011loop</ithink>\n<ioutput>if current_score > max_score:\n            max_score = current_score</ioutput>\n</iconv>\n<iconv>\n<ithink>In the findMaximumScore function we need to iterate through the entire input list to locate the highest value. The line `index += 1` advances the loop counter, ensuring we move forward one element at a time in the while\u2011loop. Without this increment, the loop would never terminate, and the function would either run forever or only examine the first element. Therefore this line is essential to correctly traverse the nums array and to complete the linear\u2011time search logic.</ithink>\n<ioutput>index += 1</ioutput>\n</iconv>\n<iconv>\n<ithink>In the findMaximumScore function, after iterating through the entire nums array, we need to output the result of the computation. The variable max_score has been maintained throughout the loop as the highest value seen so far. Returning max_score provides the caller with the final answer, completing the function's purpose of delivering the maximum value of the input list. This return statement is essential to fulfill the algorithm's contract and ensures the function's output matches the expected specification.</ithink>\n<ioutput>return max_score\n```</ioutput>\n</iconv><|role_end|><|endoftext|>", LMStyle.iCoder))
    print(extract_code(
          "\n<iconv>\n<ithink>The program needs to compute the maximum product of digits, which requires determining the number of digits in each number. To do this efficiently, the code should import the math module, specifically the log10 function, which provides a fast way to calculate the number of digits of a positive integer. Including the import at the top of the file ensures that the subsequent count_digits function can use math.log10 without errors, keeping the implementation simple and consistent with the rest of the program.</ithink>\n<ioutput>```python\n# Code here\nimport math</ioutput>\n</iconv>\n<iconv>\n<ithink>The program needs a way to determine how many digits each number has so it can compute the product of digits correctly. After reading the input list, the next logical step is to define a helper that returns the count of digits for a given integer. This function will be called for every number in the list to obtain the digit count, which is then used to calculate the product of digits. Therefore, this segment defines the count_digits function, handling the edge case of zero and using math.log10 for positive numbers to compute the digit count efficiently.</ithink>\n<ioutput>def count_digits(n):\n    if n == 0:\n        return 1\n    return int(math.log10(n)) + 1</ioutput>\n</iconv>\n<iconv>\n<ithink>The program needs a way to compute the product of all digits in the input list. After defining the helper that counts digits, the next logical step is to create a function that iterates over each number, obtains its digit count using the helper, and multiplies those counts together. This function will be called in the main block to produce the final result, so it must be defined here to keep the code modular and readable.</ithink>\n<ioutput>def product_of_digits(numbers):\n    product = 1\n    for num in numbers:\n        product *= count_digits(num)\n    return product</ioutput>\n</iconv>\n<iconv>\n<ithink>The main block needs to obtain the list of numbers to process, so it reads a line from stdin, splits it on whitespace, and converts each token to an integer. This prepares the data for the subsequent product_of_digits call, which expects a list of integers. Generating this code ensures the program can ingest input from the command line or file as required by the specification.</ithink>\n<ioutput>if __name__ == \"__main__\":\n    numbers = list(map(int, input().split()))</ioutput>\n</iconv>\n<iconv>\n<ithink>The main block has read a line of input into a list of integers called `numbers`. The next logical step is to compute the product of the digit counts of each number. This line calls the helper function `product_of_digits` with the collected list, stores the result in `result`, and then prints it. It ties together the input handling and the core computation, ensuring the program outputs the required product of digit counts.</ithink>\n<ioutput>result = product_of_digits(numbers)\n    print(result)\n```</ioutput>\n</iconv><|role_end|><|endoftext|>",
        LMStyle.iCoder))