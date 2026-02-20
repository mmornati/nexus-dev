import timeit
import dataclasses
import random
import string

@dataclasses.dataclass
class TextContent:
    text: str

def generate_data(n):
    """Generate a list of N objects with a .text attribute or a dict with 'text' key."""
    data = []
    for i in range(n):
        text = ''.join(random.choices(string.ascii_letters, k=10))
        if i % 2 == 0:
            data.append(TextContent(text=text))
        else:
            data.append({"text": text})
    return data

def naive_concat(content_list):
    text_content = ""
    for content in content_list:
        if hasattr(content, "text"):
            text_content += content.text
        elif isinstance(content, dict) and "text" in content:
            text_content += content["text"]
    return text_content

def optimized_concat(content_list):
    text_content_parts = []
    for content in content_list:
        if hasattr(content, "text"):
            text_content_parts.append(content.text)
        elif isinstance(content, dict) and "text" in content:
            text_content_parts.append(content["text"])
    return "".join(text_content_parts)

def run_benchmark():
    print("Benchmark: String Concatenation vs List Join")
    print("-" * 50)
    print(f"{'N Items':<10} | {'Naive (sec)':<15} | {'Optimized (sec)':<15} | {'Speedup':<10}")
    print("-" * 50)

    for n in [1000, 10000, 50000, 100000]:
        data = generate_data(n)

        # Time naive
        naive_time = timeit.timeit(lambda: naive_concat(data), number=10)

        # Time optimized
        optimized_time = timeit.timeit(lambda: optimized_concat(data), number=10)

        speedup = naive_time / optimized_time if optimized_time > 0 else float('inf')

        print(f"{n:<10} | {naive_time:.6f}          | {optimized_time:.6f}          | {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
