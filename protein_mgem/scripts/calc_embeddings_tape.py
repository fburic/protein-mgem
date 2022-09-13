import argparse

import tape.main as tape_main


def extended_create_embed_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Embed a set of proteins wiht a pretrained model',
        parents=[base_parser])
    parser.add_argument('data_file', type=str,
                        help='File containing set of proteins to embed')
    parser.add_argument('out_file', type=str,
                        help='Name of output file')
    parser.add_argument('from_pretrained', type=str,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--full_sequence_embed', action='store_true',
                        help='If true, saves an embedding at every amino acid position '
                             'in the sequence. Note that this can take a large amount '
                             'of disk space.')
    parser.add_argument('--gradient_accumulation_steps',
                        required=False,
                        default=1, type=int,
                        help='Batch size')

    parser.set_defaults(task='embed')
    return parser


# Rebinding works with tape-proteins 0.4
tape_main.create_embed_parser = extended_create_embed_parser
tape_main.run_embed()
