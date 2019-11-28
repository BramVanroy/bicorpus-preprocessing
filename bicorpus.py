from pathlib import Path


def construct(fsrc, ftgt, *, sep='\t', encoding='utf-8', **kwargs):
    pfsrc = Path(fsrc)
    pftgt = Path(ftgt)
    src_ext = pfsrc.suffix
    tgt_ext = pftgt.suffix
    pfout = pfsrc.with_suffix(f".{src_ext[1:]}-{tgt_ext[1:]}")

    with pfsrc.open(encoding=encoding) as fhsrc, \
            pftgt.open(encoding=encoding) as fhtgt, \
            pfout.open('w', encoding=encoding) as fhout:
        for src_line, tgt_line in zip(fhsrc, fhtgt):
            src_line = src_line.rstrip()
            tgt_line = tgt_line.rstrip()
            fhout.write(f"{src_line}{sep}{tgt_line}\n")

    print(f"Files processed as '{pfout}'")


def deconstruct(fin, *, src_ext, tgt_ext, sep='\t', encoding='utf-8', **kwargs):
    pfin = Path(fin)
    pfsrc = pfin.with_suffix(f".{src_ext}")
    pftgt = pfin.with_suffix(f".{tgt_ext}")

    with pfin.open(encoding=encoding) as fhin, \
            pfsrc.open('w', encoding=encoding) as fhsrc, \
            pftgt.open('w', encoding=encoding) as fhtgt:
        for line in fhin:
            line = line.rstrip()
            src, tgt = line.split(sep, maxsplit=2)
            fhsrc.write(f"{src}\n")
            fhtgt.write(f"{tgt}\n")

    print(f"File processed as '{pfsrc}' and '{pftgt}")


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser(description='Construct or deconstruct a bilingual corpus.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = cparser.add_subparsers(help='Construct or deconstruct')
    cparser.add_argument('--sep', default='\t', help='Separator between source and target sentence')
    cparser.add_argument('--encoding', default='utf-8', help='Encoding of input and output files')

    construct_parser = subparsers.add_parser('construct')
    construct_parser.set_defaults(func=construct)
    construct_parser.add_argument('fsrc', help='Source file')
    construct_parser.add_argument('ftgt', help='Target file')

    deconstruct_parser = subparsers.add_parser('deconstruct')
    deconstruct_parser.set_defaults(func=deconstruct)
    deconstruct_parser.add_argument('fin', help='Input file')
    deconstruct_parser.add_argument('src_ext', help='Extension for the source file')
    deconstruct_parser.add_argument('tgt_ext', help='Extension for the target file')

    cargs = cparser.parse_args()
    cargs.func(**vars(cargs))
