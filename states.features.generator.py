with open('states.txt', 'r') as f:
    output = ''
    for i in f:
        output = output + '\'' + i.strip() + '\','
    print(output)
