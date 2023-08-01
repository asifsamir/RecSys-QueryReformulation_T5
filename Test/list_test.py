import ast

ground_truths = '''[
            "framework/bundles/org.eclipse.ecf.provider/src/org/eclipse/ecf/provider/generic/ClientSOContainer.java",
            "framework/bundles/org.eclipse.ecf.provider/src/org/eclipse/ecf/provider/generic/ClientSOContainer.java"
        ]'''

ground_truths = ast.literal_eval(ground_truths)

print('--------a----------')

print(ground_truths)
print(type(ground_truths))

print('--------b----------')

# iterates over the list
for val in ground_truths:
    print(val)
    print(type(val))