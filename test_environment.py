"""
Script para testar se o ambiente Python está configurado corretamente
"""

import sys

REQUIRED_PYTHON = "python3"

def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Versão Python não reconhecida: {}".format(REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "Este projeto requer Python {}. Encontrado: Python {}".format(
                required_major, sys.version
            )
        )
    else:
        print(">>> Ambiente Python configurado corretamente!")

if __name__ == '__main__':
    main()
