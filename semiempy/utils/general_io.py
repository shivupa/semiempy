
program_name_art = """                _ ___       ___         \n  ___ ___ _ __ (_) __|_ __ | _ \_  _    \n (_-</ -_) '  \| | _|| '  \|  _/ || |   \n /__/\___|_|_|_|_|___|_|_|_|_|  \_, |   \n                                |__/    \n
"""

def print_header():
    for i in program_name_art.splitlines():
        print("{:^79}".format(i))
    print()
    print("{:^79}".format("AUTHORS : See authors.md"))
    print()
    print()
    
