from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import time
import io
import base64
import tracemalloc

app = Flask(__name__)

def dpBottomUp(weights, values, mx_weight):
    n = len(weights)
    dp = [[0 for x in range(mx_weight + 1)] for x in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, mx_weight + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w - weights[i-1]] + values[i-1], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][mx_weight]

def dpTopDown(weights, values, mx_weight):
    dp = [[-1 for _ in range(mx_weight + 1)] for _ in range(len(weights) + 1)]
    def dpTopDownHelper(i, w):
        if i == 0 or w == 0:
            return 0
        if dp[i][w] != -1:
            return dp[i][w]
        if weights[i-1] <= w:
            dp[i][w] = max(dpTopDownHelper(i-1, w - weights[i-1]) + values[i-1], dpTopDownHelper(i-1, w))
            return dp[i][w]
        else:
            dp[i][w] = dpTopDownHelper(i-1, w)
            return dp[i][w]
    return dpTopDownHelper(len(weights), mx_weight)

def bruteForce(weights, values, mx_weight):
    n = len(weights)
    max_value = 0
    for i in range(1 << n):
        subset_weight = 0
        subset_value = 0
        for j in range(n):
            if i & (1 << j):
                subset_weight += weights[j]
                subset_value += values[j]
        if subset_weight <= mx_weight and subset_value > max_value:
            max_value = subset_value
    return max_value

def greedy(weights, values, mx_weight):
    combined = [(v, w) for v, w in zip(values, weights)]
    combined.sort(reverse=True, key=lambda x: x[0] / x[1])
    current_weight = 0
    max_value = 0
    for value, weight in combined:
        if current_weight + weight <= mx_weight:
            current_weight += weight
            max_value += value
    return max_value

def measure_time(func, weights, values, mx_weight):
    start_time = time.time()
    func(weights, values, mx_weight)
    return time.time() - start_time

def measure_space(func, weights, values, mx_weight):
    tracemalloc.start()
    func(weights, values, mx_weight)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 10**6  # Convert to MB

def get_complexity(algorithm):
    complexities = {
        'dpBottomUp': ('O(n*W)', 'O(n*W)'),
        'dpTopDown': ('O(n*W)', 'O(n*W)'),
        'bruteForce': ('O(2^n)', 'O(n)'),
        'greedy': ('O(n log n)', 'O(1)')
    }
    return complexities.get(algorithm, ('Unknown', 'Unknown'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    weights = list(map(int, request.form['weights'].split(',')))
    values = list(map(int, request.form['values'].split(',')))
    max_weight = int(request.form['max_weight'])
    algorithm = request.form['algorithm']
    
    if algorithm == 'dpBottomUp':
        result = dpBottomUp(weights, values, max_weight)
    elif algorithm == 'dpTopDown':
        result = dpTopDown(weights, values, max_weight)
    elif algorithm == 'bruteForce':
        result = bruteForce(weights, values, max_weight)
    elif algorithm == 'greedy':
        result = greedy(weights, values, max_weight)
    else:
        result = 'Invalid algorithm selected'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # Adjust figsize here
    graph_url, result_text = generate_graph(weights, values, max_weight, algorithm, ax1, ax2, result)
    time_complexity, space_complexity = get_complexity(algorithm)
    result_text += f' | Time Complexity: {time_complexity} | Space Complexity: {space_complexity}'
    return render_template('index.html', graph_url=graph_url, result=result_text)

def generate_graph(weights, values, max_weight, selected_algorithm, ax1, ax2, result):
    if weights is None:
        weights = list(range(1, 21))
    if values is None:
        values = list(range(1, 21))
    if max_weight is None:
        max_weight = 50
    
    algorithms = {
        'dpBottomUp': dpBottomUp,
        'dpTopDown': dpTopDown,
        'bruteForce': bruteForce,
        'greedy': greedy
    }
    
    times = {name: [] for name in algorithms}
    spaces = {name: [] for name in algorithms}
    input_sizes = list(range(1, 21))
    
    for size in input_sizes:
        test_weights = weights[:size]
        test_values = values[:size]
        
        for name, func in algorithms.items():
            exec_time = measure_time(func, test_weights, test_values, max_weight)
            exec_space = measure_space(func, test_weights, test_values, max_weight)
            times[name].append(exec_time)
            spaces[name].append(exec_space)
    
    for name, time_values in times.items():
        if name == selected_algorithm:
            ax1.plot(input_sizes, time_values, label=name, linewidth=2.5)
        else:
            ax1.plot(input_sizes, time_values, label=name, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Input Size (number of items)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Algorithm Time Complexity Comparison')
    ax1.legend()
    ax1.grid(True)
    
    for name, space_values in spaces.items():
        if name == selected_algorithm:
            ax2.plot(input_sizes, space_values, label=name, linewidth=2.5)
        else:
            ax2.plot(input_sizes, space_values, label=name, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Input Size (number of items)')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Algorithm Space Complexity Comparison')
    ax2.legend()
    ax2.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_url = 'data:image/png;base64,{}'.format(base64.b64encode(buf.getvalue()).decode('utf8'))
    buf.close()
    
    result_text = 'Result: {}'.format(result)
    
    return graph_url, result_text

if __name__ == '__main__':
    app.run(debug=True)
