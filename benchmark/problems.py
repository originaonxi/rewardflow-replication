"""
20 multi-step reasoning problems.
Each has a known correct answer AND known correct intermediate steps.
This lets us measure whether RewardFlow correctly identifies
which steps were on the path to success.

Key: we know the GROUND TRUTH step quality for each problem.
We can therefore measure:
- Did sparse reward identify the right steps? (baseline)
- Did RewardFlow identify the right steps? (our method)
"""

PROBLEMS = [
    {
        "id": "MATH_001",
        "problem": "A train leaves city A at 60 mph. Another leaves city B (300 miles away) toward A at 90 mph. When do they meet, and how far from A?",
        "answer": "2 hours, 120 miles from A",
        "answer_value": 120.0,
        "correct_steps": [
            "Combined speed = 60 + 90 = 150 mph",
            "Time to meet = 300 / 150 = 2 hours",
            "Distance from A = 60 * 2 = 120 miles"
        ],
        "n_steps": 3,
        "category": "word_problem",
    },
    {
        "id": "MATH_002",
        "problem": "Find the sum of interior angles of a polygon with 8 sides.",
        "answer": "1080 degrees",
        "answer_value": 1080.0,
        "correct_steps": [
            "Formula: (n-2) * 180 degrees",
            "n = 8, so (8-2) * 180",
            "6 * 180 = 1080 degrees"
        ],
        "n_steps": 3,
        "category": "geometry",
    },
    {
        "id": "MATH_003",
        "problem": "A bacteria culture doubles every 3 hours. Starting with 100 bacteria, how many after 12 hours?",
        "answer": "1600",
        "answer_value": 1600.0,
        "correct_steps": [
            "Number of doublings = 12/3 = 4",
            "Final count = 100 * 2^4",
            "100 * 16 = 1600"
        ],
        "n_steps": 3,
        "category": "exponential",
    },
    {
        "id": "MATH_004",
        "problem": "If 3x + 7 = 4x - 5, find x. Then find x squared.",
        "answer": "x=12, x squared=144",
        "answer_value": 144.0,
        "correct_steps": [
            "3x + 7 = 4x - 5",
            "7 + 5 = 4x - 3x",
            "x = 12",
            "x squared = 144"
        ],
        "n_steps": 4,
        "category": "algebra",
    },
    {
        "id": "MATH_005",
        "problem": "A store has 40% off sale. After the discount, there's an additional 15% off. What's the total effective discount?",
        "answer": "49%",
        "answer_value": 49.0,
        "correct_steps": [
            "After 40% off: price = 0.60 * original",
            "After additional 15% off: 0.85 * 0.60 = 0.51",
            "Total discount = 1 - 0.51 = 0.49 = 49%"
        ],
        "n_steps": 3,
        "category": "percentage",
    },
    {
        "id": "MATH_006",
        "problem": "How many ways can 4 people be arranged in a line? What if two specific people must not be adjacent?",
        "answer": "24 total, 12 non-adjacent",
        "answer_value": 12.0,
        "correct_steps": [
            "Total arrangements = 4! = 24",
            "Adjacent arrangements: treat pair as unit = 3! * 2 = 12",
            "Non-adjacent = 24 - 12 = 12"
        ],
        "n_steps": 3,
        "category": "combinatorics",
    },
    {
        "id": "MATH_007",
        "problem": "A rectangular garden is 3 times as long as it is wide. If the perimeter is 96m, what is the area?",
        "answer": "432 sq m",
        "answer_value": 432.0,
        "correct_steps": [
            "Let width = w, length = 3w",
            "Perimeter: 2(w + 3w) = 96, so 8w = 96, w = 12",
            "Length = 36, Area = 12 * 36 = 432"
        ],
        "n_steps": 3,
        "category": "geometry",
    },
    {
        "id": "MATH_008",
        "problem": "If log base 2 of 8 plus log base 2 of x equals log base 2 of 64, find x.",
        "answer": "8",
        "answer_value": 8.0,
        "correct_steps": [
            "log2(8) = 3",
            "3 + log2(x) = log2(64) = 6",
            "log2(x) = 3, so x = 2^3 = 8"
        ],
        "n_steps": 3,
        "category": "logarithm",
    },
    {
        "id": "MATH_009",
        "problem": "A car depreciates 20% per year. After 3 years, what fraction of original value remains?",
        "answer": "0.512 or 51.2%",
        "answer_value": 0.512,
        "correct_steps": [
            "After year 1: 0.80 of original",
            "After year 2: 0.80^2 = 0.64",
            "After year 3: 0.80^3 = 0.512"
        ],
        "n_steps": 3,
        "category": "exponential",
    },
    {
        "id": "MATH_010",
        "problem": "Two pipes fill a tank in 4 and 6 hours. A drain empties it in 12 hours. How long to fill with all open?",
        "answer": "4 hours",
        "answer_value": 4.0,
        "correct_steps": [
            "Pipe A rate: 1/4 per hour",
            "Pipe B rate: 1/6 per hour",
            "Drain rate: -1/12 per hour",
            "Net rate: 1/4 + 1/6 - 1/12 = 3/12 + 2/12 - 1/12 = 4/12 = 1/3",
            "Time = 1/(1/3) = 3 hours"
        ],
        "n_steps": 5,
        "category": "rates",
    },
    {
        "id": "MATH_011",
        "problem": "Find the 15th term of arithmetic sequence: 3, 7, 11, 15...",
        "answer": "59",
        "answer_value": 59.0,
        "correct_steps": [
            "Common difference d = 7-3 = 4",
            "Formula: a_n = a_1 + (n-1)d",
            "a_15 = 3 + 14*4 = 3 + 56 = 59"
        ],
        "n_steps": 3,
        "category": "sequences",
    },
    {
        "id": "MATH_012",
        "problem": "A 10% salt solution is mixed with 30% salt solution to get 20 liters of 15% solution. How much of each?",
        "answer": "15 liters of 10%, 5 liters of 30%",
        "answer_value": 15.0,
        "correct_steps": [
            "Let x = liters of 10% solution",
            "0.10x + 0.30(20-x) = 0.15*20",
            "0.10x + 6 - 0.30x = 3",
            "-0.20x = -3, x = 15"
        ],
        "n_steps": 4,
        "category": "mixture",
    },
    {
        "id": "MATH_013",
        "problem": "What is the probability of getting at least one head in 4 coin flips?",
        "answer": "15/16",
        "answer_value": 15/16,
        "correct_steps": [
            "P(no heads) = (1/2)^4 = 1/16",
            "P(at least one head) = 1 - 1/16 = 15/16"
        ],
        "n_steps": 2,
        "category": "probability",
    },
    {
        "id": "MATH_014",
        "problem": "A ladder 10m long leans against a wall. Base is 6m from wall. How high does it reach? If base slides to 8m, how far does top slide down?",
        "answer": "8m high initially, slides down 2m",
        "answer_value": 2.0,
        "correct_steps": [
            "Initial height: sqrt(10^2-6^2) = sqrt(100-36) = sqrt(64) = 8m",
            "New height: sqrt(10^2-8^2) = sqrt(100-64) = sqrt(36) = 6m",
            "Slide = 8 - 6 = 2m"
        ],
        "n_steps": 3,
        "category": "geometry",
    },
    {
        "id": "MATH_015",
        "problem": "Simple interest on $5000 at 8% for 3 years vs compound interest. What's the difference?",
        "answer": "$97.12",
        "answer_value": 97.12,
        "correct_steps": [
            "Simple: 5000 * 0.08 * 3 = $1200",
            "Compound: 5000 * (1.08)^3 - 5000 = 5000 * 1.2597 - 5000 = $1298.56",
            "Difference = 1298.56 - 1200 = $98.56"
        ],
        "n_steps": 3,
        "category": "finance",
    },
    {
        "id": "MATH_016",
        "problem": "Find the area of triangle with sides 5, 12, 13.",
        "answer": "30 sq units",
        "answer_value": 30.0,
        "correct_steps": [
            "Check: 5^2 + 12^2 = 25 + 144 = 169 = 13^2 -- right triangle",
            "Area = (1/2) * base * height = (1/2) * 5 * 12 = 30"
        ],
        "n_steps": 2,
        "category": "geometry",
    },
    {
        "id": "MATH_017",
        "problem": "If f(x) = 2x^2 - 3x + 1, find f(f(2)).",
        "answer": "f(f(2)) = f(3) = 10",
        "answer_value": 10.0,
        "correct_steps": [
            "f(2) = 2(4) - 3(2) + 1 = 8 - 6 + 1 = 3",
            "f(f(2)) = f(3) = 2(9) - 3(3) + 1 = 18 - 9 + 1 = 10"
        ],
        "n_steps": 2,
        "category": "functions",
    },
    {
        "id": "MATH_018",
        "problem": "How many 3-digit numbers are divisible by both 4 and 6?",
        "answer": "75",
        "answer_value": 75.0,
        "correct_steps": [
            "LCM(4,6) = 12",
            "Smallest 3-digit multiple of 12: 108",
            "Largest: 996",
            "Count = (996-108)/12 + 1 = 888/12 + 1 = 74 + 1 = 75"
        ],
        "n_steps": 4,
        "category": "number_theory",
    },
    {
        "id": "MATH_019",
        "problem": "A sphere has surface area 100*pi cm^2. What is its volume?",
        "answer": "500*pi/3 approx 523.6 cm^3",
        "answer_value": 523.6,
        "correct_steps": [
            "Surface area = 4*pi*r^2, so 4*pi*r^2 = 100*pi",
            "r^2 = 25, r = 5",
            "Volume = (4/3)*pi*r^3 = (4/3)*pi*(125) = 500*pi/3 approx 523.6"
        ],
        "n_steps": 3,
        "category": "geometry",
    },
    {
        "id": "MATH_020",
        "problem": "Workers A and B together finish a job in 6 days. A alone takes 10 days. B alone takes how long? If C joins and they finish in 4 days, how long for C alone?",
        "answer": "B=15 days, C=20 days",
        "answer_value": 20.0,
        "correct_steps": [
            "A's rate = 1/10, Combined = 1/6",
            "B's rate = 1/6 - 1/10 = 5/30 - 3/30 = 2/30 = 1/15, B=15 days",
            "With C: 1/10 + 1/15 + 1/C = 1/4",
            "1/C = 1/4 - 1/10 - 1/15 = 15/60 - 6/60 - 4/60 = 5/60 = 1/12",
            "C = 12 days"
        ],
        "n_steps": 5,
        "category": "rates",
    },
]
