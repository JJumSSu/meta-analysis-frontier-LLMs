import json
import numpy as np
import matplotlib.pyplot as plt


FEW_SHOT_RULE_DICT = {
    'Zero-shot': 0, 
    'few-shot': "xx",
    'all': "xx", 
    'Few-shot': "xx", 
    'k': "xx",
    'zero': 0,
    'Variable': "xx",
    'few': "xx",
    'Zero': 0,
    'sc_5': "xx",
    'Zero-shot': 0,
}


def process_ds(dataset, arxiv_id_to_venue, venue_filter, process_few_shot=False):
    target_metrics = ['Exact Match', 'Precision', 'Accuracy', 'BLEU', 'Rouge', 'F1', 'Recall']
    dataset = dataset.filter(lambda x: x['metric'] in target_metrics)

    if process_few_shot:
        number_of_shots = dataset['number_of_shots']
        number_of_shots_converted = []
        for _, shot in enumerate(number_of_shots):
            if shot in FEW_SHOT_RULE_DICT.keys():
                shot = FEW_SHOT_RULE_DICT[shot]
            if shot != 'xx':
                try:
                    number_of_shots_converted.append(str(shot))
                except ValueError:
                    number_of_shots_converted.append("xx")
            else:
                number_of_shots_converted.append("xx")
        
        dataset = dataset.remove_columns("number_of_shots")
        dataset = dataset.add_column('number_of_shots', number_of_shots_converted)
        dataset = dataset.filter(lambda x: x['number_of_shots'] != "xx")
        dataset = dataset.map(lambda x: {'number_of_shots': int(x['number_of_shots']) 
                                        if x['number_of_shots'].isdigit() else x['number_of_shots']})

    if venue_filter:
        with open(arxiv_id_to_venue, 'r') as f:
            arxiv_id_to_venue = json.load(f)

        venues = []
        for instance in dataset:
            venue_infomration = arxiv_id_to_venue[instance['table_source_arxiv_id']]
            venues.append(venue_infomration)
        
        dataset = dataset.add_column("venues", venues)
        dataset = dataset.filter(lambda x: x['venues'] != "CoRR")

    return dataset


def create_boxplots(total_deltas_per_category, total_deltas_aggregated_by_paper_per_category, total_median, desired_order, output_path, prefix='CoT', colors=['#695CDB', '#78A8E6']):
    
    plt.figure(figsize=(6, 4)) if len(desired_order) < 6 else plt.figure(figsize=(6, 6))

    ordered_keys = [key for key in desired_order if key in total_deltas_per_category]
    total_deltas_per_category = {key: total_deltas_per_category[key] for key in ordered_keys}
    total_deltas_per_category = {k: total_deltas_per_category[k] for k in reversed(total_deltas_per_category)}  

    y_pos = 0
    y_ticks = []
    for category, data in total_deltas_per_category.items():  
        y_ticks.append(category)
        category_average = np.mean(data)

        plt.plot(category_average, y_pos, marker='*', color=colors[0], markersize=10, zorder=3)

        bp = plt.boxplot(data, positions=[y_pos], vert=False,
                widths=0.5, patch_artist=True, showfliers=False)
    
        plt.setp(bp['boxes'], facecolor='lightgray', alpha=0.6)
        plt.setp(bp['medians'], color='black')  
    
        scatter_y = np.full_like(data, y_pos)
        plt.scatter(data, scatter_y, color="darkgray", s=3)  

        paper_data = []
        for k in total_deltas_aggregated_by_paper_per_category[category].keys():
            paper_data.append(total_deltas_aggregated_by_paper_per_category[category][k])
        scatter_y = np.full_like(paper_data, y_pos)
        plt.scatter(paper_data, scatter_y, color=colors[1], alpha=0.8, s=7)
    
        y_pos += 1  
    
    plt.axvline(x=total_median, color='red', linestyle='--', label=f'Median {prefix} Delta: {round(total_median,1)}', alpha=0.5)
    
    plt.yticks(range(len(total_deltas_per_category)), y_ticks, fontsize=12)
    plt.xlabel('Performance Delta', fontsize=12)
    plt.xlim(-15, 45)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
