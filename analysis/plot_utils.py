import matplotlib.pyplot as plt
import numpy as np

def plot_question_types_accuracy(llava=False):
    file_path = 'data/question_types_accuracy_llava.txt'
    if not llava:
        file_path = 'data/question_types_accuracy_prophet.txt'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    question_types = []
    accuracies = []
    len_questions = []
    for line in lines:
        question_type, accuracy_list = line.strip().split(':')
        question_types.append(question_type.replace(' type of questions', ''))
        questions_len, accuracy = accuracy_list.split(',')
        accuracies.append(float(accuracy.replace('%', '')))
        len_questions.append(int(questions_len))
    
    y_pos = np.arange(len(question_types))
    # rotate the x-axis labels by 45 degrees
    
    plt.bar(y_pos, accuracies, align='center', alpha=0.5)
    plt.xticks(y_pos, question_types, rotation=45)
    plt.ylabel('Accuracy (%)')
    if llava:
        plt.title('LLaVA Accuracy for each question type')
    else:
        print('Prophet')
        plt.title('Prophet Accuracy for each question type')
    # Set y-axis to range from 0 to 100
    plt.ylim(0, 100)
    
    # Loop to add len_questions above each bar
    for i in range(len(y_pos)):
        plt.text(y_pos[i], accuracies[i], str(len_questions[i]), ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.show()
    plt.savefig(file_path.replace('.txt', '.png'))
    

if __name__ == "__main__":
    # plot_question_types_accuracy(llava=True)
    
    accuracies = [30.1, 44.7, 47.2, 50.9, 51.4]
    answer_aware_examples = [0, 1, 8, 16, 20]
    
    y_pos = np.arange(len(accuracies))
    plt.plot(y_pos, accuracies, marker='o', linestyle=':', color='r') # Changed to dotted line with linestyle=':'
    
    # rotate the x-axis labels by 45 degrees
    plt.xticks(y_pos, answer_aware_examples, rotation=45) # x-axis labels changed to answer_aware_examples
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Number of in-context examples')
    plt.title('Prophet: IC Example Count vs Accuracy Plot')
    
    plt.grid(True) # Adding grid lines

    plt.tight_layout()
    plt.show()
    plt.savefig('data/prophet_accuracy_answer_aware.png')