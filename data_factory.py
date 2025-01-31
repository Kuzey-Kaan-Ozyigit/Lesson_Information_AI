import random

training_data = 1000

with open("student_scores.txt", "w") as file:
    for _ in range(training_data):
        a, b, c = random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)
        file.write(f"{a},{b},{c}\n")

with open("pass_fail_labels.txt", "w") as file:
    with open("student_scores.txt", "r") as value:
        for line in value:
            numbers = list(map(int, line.strip().split(",")))
            average = sum(numbers) / len(numbers)
            if average >= 50:
                data = 1
            else:
                data = 0
            file.write(f"{data}\n")

with open("category_labels.txt", "w") as file:
    with open("student_scores.txt", "r") as value:
        for line in value:
            numbers = list(map(int, line.strip().split(",")))
            average = sum(numbers) / len(numbers)
            if average >= 0 and average <=49:
                data_1 = 1
                data_2 = 0
                data_3 = 0
                data_4 = 0
            elif average >= 50 and average <=64:
                data_1 = 0
                data_2 = 1
                data_3 = 0
                data_4 = 0
            elif average >= 65 and average <=84:
                data_1 = 0
                data_2 = 0
                data_3 = 1
                data_4 = 0
            elif average >= 85 and average <=100:
                data_1 = 0
                data_2 = 0
                data_3 = 0
                data_4 = 1
            file.write(f"{data_1},{data_2},{data_3},{data_4}\n")