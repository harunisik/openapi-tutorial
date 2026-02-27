import taskmanager

def main():
    a = [1, 2, 3]

    b = a[1:]
    b.append(4)

    a.append(5)

    c = [1, 2, 3, 4, 5]
    print(f"c: {c[1:-1]}")

    print(f"a: {a}")
    print(f"b: {b}")

    print(taskmanager.Task)

if __name__ == "__main__":
    main()
