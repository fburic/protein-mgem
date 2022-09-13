import pandas as pd


from scripts.general.attribution import dist_pattern_structure, load_patterns
from scripts.general import util


def main():
    logger = util.get_logger()
    args = util.get_args()
    # FIXME: Structural and notebook producing it are missing
    secondary_struct = pd.read_csv(args.exp_dir / 'secondary_struct.csv')

    logger.info('Loading patterns ...')
    secondary_struct['pattern_list'] = secondary_struct['swissprot_ac'].map(
        lambda prot_id: load_patterns(prot_id, args.exp_dir)
    )
    secondary_struct = secondary_struct.explode('pattern_list')
    secondary_struct[['attention_pattern', 'focus_pattern']] = pd.DataFrame(
        secondary_struct['pattern_list'].tolist(),
        index=secondary_struct.index
    )
    secondary_struct = secondary_struct[~secondary_struct['attention_pattern'].isna()]
    secondary_struct = secondary_struct.drop(columns='pattern_list')
    logger.info('Done.')

    logger.info('Calculating Hamming distance between patterns and structures...')
    secondary_struct = secondary_struct.assign(
        dist_attention_structure = secondary_struct.apply(
            lambda row: dist_pattern_structure(row['attention_pattern'],
                                               row['secondary_structure']),
            axis='columns')
    )
    secondary_struct = secondary_struct.assign(
        dist_focus_structure = secondary_struct.apply(
            lambda row: dist_pattern_structure(row['focus_pattern'],
                                               row['secondary_structure']),
            axis='columns')
    )
    secondary_struct.to_csv(args.exp_dir / 'secondary_struct_match.csv', index=False)
    logger.info('Done.')





if __name__ == '__main__':
    main()
