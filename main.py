# Dina Englard

import argparse
import random
import ratings
import matplotlib.pyplot as plt


POPULATION_SIZE = 200
MAX_GENERATION_NUM = 600
ELITE_GROUP_SIZE = 5
ACCEPTABLE_GRADE = 8
MUTATION_RATE = 0.1
CONVERGENCE = 50
alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
rating_dictionary = ratings.trigrams
function_count = 0
lamarck = 0
darwin = 1
SWAPS = 2


with open('dict.txt', 'r') as f:
    valid_words = set(word.strip().lower() for word in f)


def generate_random_permutation(encoded_text):
    perm = list(alphabet)
    random.shuffle(perm)
    global lamarck
    if lamarck:
        for i in range(SWAPS):
            new_child = optimize(perm, encoded_text)[0]
            perm = list(new_child)

    return perm


# turn text into long string of uppercase letters
def extract_letters(text):
    letters = ''.join(c for c in text if c.isalpha())
    return letters.upper()


def translate(perm, enc_text):
    decoded_text = enc_text.translate(str.maketrans(''.join(perm), 'abcdefghijklmnopqrstuvwxyz'))
    return decoded_text


def init(enc_text):
    # create a random perm of alphabet
    population = {}
    while len(population) < POPULATION_SIZE:
        # create random perm
        perm = generate_random_permutation(enc_text)
        decoded_text = translate(perm, enc_text)
        score = fitness(decoded_text)
        population[''.join(perm)] = score
        # print(population, len(population))
    return population


def trigram_score(raw_text):
    text = extract_letters(raw_text)
    rating = 0
    global function_count
    function_count += 1
    # print(function_count, generation, individual_num)
    for window_right_index in range(3, len(text)):
        trigramStr = text[window_right_index - 3:window_right_index]
        if trigramStr in rating_dictionary:
            rating += rating_dictionary[trigramStr]
    return rating


def word_count_score(raw_text):
    global valid_words
    decoded_words = raw_text.lower().split()
    # for word in decoded_words:
    #     if word not in valid_words:
    #         print(word)
    valid_word_count = sum(word in valid_words for word in decoded_words)

    return valid_word_count / len(decoded_words)


def fitness(raw_text):
    # return 4* word_count_score(raw_text)+trigram_score(raw_text)
    return trigram_score(raw_text)


def crossover(parent1, parent2, encoded_text):
    child = list(parent1)
    indices = sorted(random.sample(range(len(alphabet)), 2))
    for i in range(indices[0], indices[1]):
        child[i] = parent2[i]
    repaired_child = repair_key(child)
    mutated_child = mutate(repaired_child)
    return mutated_child


def optimize(perm, encoded_text):
    rand1 = random.randint(0, 25)
    rand2 = random.randint(0, 25)
    # swapping same index would be pointless
    while rand2 == rand1:
        rand2 = random.randint(0, 25)
    new_perm = list(perm)
    new_perm[rand1], new_perm[rand2] = perm[rand2], perm[rand1]
    decoded_text1 = translate(new_perm, encoded_text)
    decoded_text2 = translate(perm, encoded_text)
    new_rating = fitness(decoded_text1)
    old_rating = fitness(decoded_text2)
    if new_rating > old_rating:
        return new_perm, new_rating
    return perm, old_rating


# Remove duplicate letters from a key and fill the missing letters
def repair_key(key):
    unique_letters = list(set(key))
    missing_letters = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in unique_letters]
    replaced_key = ''
    letter_count = {}
    for letter in key:
        if letter in letter_count:
            if letter_count[letter] == 1:
                replaced_key += missing_letters.pop(0)
            else:
                replaced_key += letter
            letter_count[letter] += 1
        else:
            replaced_key += letter
            letter_count[letter] = 1

    return list(replaced_key)


def mutate(perm):
    for i in range(len(alphabet)):
        if random.random() < MUTATION_RATE:
            j = random.randint(0, len(alphabet) - 1)
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def main(enc_file):
    # Read the dictionary file and store the valid English words
    top_values = []
    worst_values = []
    avg_values = []
    for it in range(1):
        # Read the encoded file
        with open(enc_file, 'r') as f:
            encoded_text = f.read().upper()
        generation = 0
        population_dict = init(encoded_text)

        global darwin
        if darwin:
            population_with_optimizitaion_scores = population_dict.copy()
            for i in range(SWAPS):
                for key in population_with_optimizitaion_scores:
                    score2 = optimize(key, encoded_text)[1]
                    population_with_optimizitaion_scores[key] = score2
                    # sort
            population_with_optimizitaion_scores = dict(sorted(population_with_optimizitaion_scores.items(), key=lambda x: x[1], reverse=True))
            # creates population with regular perms using scores of optimized perms
            population_dict = population_with_optimizitaion_scores.copy()
        else:
            population_dict = dict(sorted(population_dict.items(), key=lambda x: x[1], reverse=True))

        # Create elite_population with the top 10 entries
        elite_population_dict = dict(list(population_dict.items())[:10])
        if darwin:
            # re-evaluate the top 10 with their actual score and sort
            for key in elite_population_dict:
                decoded_text = translate(key, encoded_text)
                score2 = fitness(decoded_text)
                elite_population_dict[key] = score2
            elite_population_dict = dict(sorted(elite_population_dict.items(), key=lambda x: x[1], reverse=True))

        top_values.append(list(elite_population_dict.values())[0])
        worst_values.append(list(population_dict.values())[-1])
        avg_values.append(sum(population_dict.values()) / len(population_dict))
        print(generation, top_values[-1], avg_values[-1], worst_values[-1], elite_population_dict)

        # while generation<MAX_GENERATION_NUM && top_value<ACCEPTABLE_GRADE,create new pop.
        j = 0
        while generation < MAX_GENERATION_NUM and top_values[-1] < ACCEPTABLE_GRADE:
            if j >= CONVERGENCE:
                print("CONVERGENCE")
                break

            # create new population starting with the elite of previous gen
            new_population_dict = elite_population_dict
            while len(new_population_dict) < POPULATION_SIZE:
                # add people to the population
                # call crossover and mutation functions: to return new perm
                # pick 2 parents randomly from elite_pop

                while True:
                    random_number1 = random.randint(0, 9)
                    random_number2 = random.randint(0, 9)
                    if random_number1 != random_number2:
                        break

                elite_population_list = list(elite_population_dict.keys())
                parent1 = elite_population_list[random_number1]
                parent2 = elite_population_list[random_number2]
                perm = crossover(parent1, parent2, encoded_text)

                global lamarck
                if lamarck:
                    for i in range(SWAPS):
                        new_child = optimize(perm, encoded_text)
                        perm = new_child[0]
                        score = new_child[1]
                else:
                    # rate each perm
                    decoded_text = translate(perm, encoded_text)
                    score = fitness(decoded_text)

                new_population_dict[''.join(perm)] = score

            # once population size is big enough:
            if darwin:
                population_with_optimizitaion_scores = new_population_dict.copy()
                for i in range(SWAPS):
                    for key in population_with_optimizitaion_scores:
                        score2 = optimize(key, encoded_text)[1]
                        population_with_optimizitaion_scores[key] = score2
                        # sort
                population_with_optimizitaion_scores = dict(
                    sorted(population_with_optimizitaion_scores.items(), key=lambda x: x[1], reverse=True))
                # creates population with regular perms using scores of optimized perms
                new_population_dict = population_with_optimizitaion_scores.copy()
            else:
                new_population_dict = dict(sorted(new_population_dict.items(), key=lambda x: x[1], reverse=True))

            # Create elite_population with the top 10 entries
            elite_population_dict = dict(list(new_population_dict.items())[:10])

            if darwin:
                # re-evaluate the top 10 with their actual score and sort
                for key in elite_population_dict:
                    decoded_text = translate(key, encoded_text)
                    score2 = fitness(decoded_text)
                    elite_population_dict[key] = score2
                elite_population_dict = dict(sorted(elite_population_dict.items(), key=lambda x: x[1], reverse=True))

            generation += 1
            print(generation, top_values[-1], avg_values[-1], worst_values[-1], elite_population_dict)

            if top_values[-1] < list(elite_population_dict.values())[0]:
                j = 0
            j += 1

            # real top value, also in darwin
            top_values.append(list(elite_population_dict.values())[0])
            worst_values.append(list(new_population_dict.values())[-1])
            avg_values.append(sum(new_population_dict.values()) / len(population_dict))


        # outside the while - means I got here if I exceeded the max_generation_number, or if my top value is greater than acc_grade
        # here create the files needed.

    # end of for loop of num_of_iterations

    elite_population_list = list(elite_population_dict.keys())
    bestperm = elite_population_list[0]

    # Save the decoded text to plain.txt
    t = translate(bestperm, encoded_text)
    with open('plain.txt', 'w') as f:
        f.write(t)

    # Save the permutation to perm.txt

    with open('perm.txt', 'w') as f:
        f.write(
            "\n".join(
                c + " " + p
                for c, p in zip(alphabet, bestperm)
            )
        )

    return top_values, worst_values, avg_values


def update_user_args():
    global lamarck
    global darwin

    parser = argparse.ArgumentParser(
        description="Genetic Algorithm for Substitution Cipher Decryption.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('type',
                        choices=['regular', 'darwin', 'lamarck'],
                        help='Choose the genetic algorithm type to execute')
    parser.add_argument('-t', '--times', help='Execute multiple runs for research', default=1, type=int)

    args = parser.parse_args()

    if args.type == 'regular':
        print("Regular Execution")
        darwin, lamarck = 0, 0
    elif args.type == 'darwin':
        print("Darwin Execution")
        darwin, lamarck = 1, 0
    elif args.type == 'lamarck':
        print("Lamarck Execution")
        darwin, lamarck = 0, 1

    return args.times


def handle_single_execution():
    top_values, worst_values, avg_values = main('enc.txt')
    print("num of calls to fitness func: " + str(function_count))

    # now I want to check how many of the words in plain.txt are in dict.txt
    with open('plain.txt', 'r') as f:
        plain = f.read()
    print("valid word count is: " + str(word_count_score(plain)))

    # Plot graph of current execution
    print("Showing progress graph")

    plt.plot(range(len(top_values)), top_values, color='g', label='best')
    plt.plot(range(len(worst_values)), worst_values, color='r', label='worst')
    plt.plot(range(len(avg_values)), avg_values, color='b', label='avg')

    plt.xlabel("generation")
    plt.ylabel("score")
    plt.title("Generation to Score graph")

    plt.legend()
    plt.show()

    # usually for over about 50% correct, it got the write perm.


def handle_multiple_executions(iterations):
    global function_count

    top_score = []
    calls_fitness = []
    words_score = []

    for i in range(iterations):
        function_count = 0      # initialize global state for next execution
        print("Executing #" + str(i+1))
        top_values, worst_values, avg_values = main('enc.txt')
        print("num of calls to fitness func: " + str(function_count))

        # now I want to check how many of the words in plain.txt are in dict.txt
        with open('plain.txt', 'r') as f:
            plain = f.read()
        w_score = word_count_score(plain)
        print("valid word count is: " + str(w_score))

        top_score.append(top_values[0])
        calls_fitness.append(function_count)
        words_score.append(w_score)

    print("Average Summary:")
    print("Average Top score: " + str(sum(top_score) / len(top_score)))
    print("Average Fitness Calls: " + str(sum(calls_fitness) / len(calls_fitness)))
    print("Average Words Score: " + str(sum(words_score) / len(words_score)))

    # Plot graph of executions summary
    print("Showing progress graphs")

    plt.plot(range(len(top_score)), top_score, color='b', label='Top Score')
    plt.xlabel("execution")
    plt.ylabel("Top Scores")
    plt.title("Executions statistics summary for top scores")
    plt.legend()
    plt.show()

    plt.plot(range(len(calls_fitness)), calls_fitness, color='b', label='Fitness Calls')
    plt.xlabel("execution")
    plt.ylabel("Fitness Calls")
    plt.title("Executions statistics summary for fitness calls")
    plt.legend()
    plt.show()

    plt.plot(range(len(words_score)), words_score, color='b', label='Words Score')
    plt.xlabel("execution")
    plt.ylabel("Words Scores")
    plt.title("Executions statistics summary for words scores")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    number_of_iterations = update_user_args()

    if number_of_iterations == 1:
        handle_single_execution()
    elif number_of_iterations > 1:
        handle_multiple_executions(number_of_iterations)
    else:
        print("Please enter positive number of executions times.")
