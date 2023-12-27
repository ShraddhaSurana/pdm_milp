from pulp import LpVariable, LpProblem, lpSum, LpBinary, LpMaximize

tests = ["Test1", "Test2", "Test3"]
equipment = ["Equipment1", "Equipment2", "Equipment3"]

prob = LpProblem("EquipmentScheduling", LpMaximize)

x = LpVariable.dicts("x", ((e, t, i) for e in equipment for t in tests for i in range(10)), cat="Binary")

for e in equipment:
    for i in range(10):  # 10 time slots
        prob += lpSum(x[e, t, i] for t in tests) <= 1  # At most one test for each equipment at each time slot

prob += lpSum(x[e, t, i] for e in equipment for t in tests for i in range(10))
