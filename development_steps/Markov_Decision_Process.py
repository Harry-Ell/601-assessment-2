from Gridworld_Constructor import Gridworld_Constructor
from MDP import GenericMDP



def Value_Iteration():
    print("\n")
    print("Welcome to the MDP Solver. Is this a GridWorld problem?", flush=True)
    problem_type_validity = False
    
    while not problem_type_validity:
        problem_type = input("Enter 'Gridworld', or 'Other':  ").strip()
        if problem_type.lower() == "gridworld":
            problem_type_validity = True
            gridworld_helper_func()
        elif problem_type.lower() == 'other':
            problem_type_validity = True
            print("Unsupported problem type.")
        else:
            print("Please check spelling of problem type.")

def gridworld_helper_func():
    """CLI for collecting all inputs required to specify a GridWorld."""
    
    # Get grid dimensions
    dimension_type_validity = False
    while not dimension_type_validity:
        print("\n")
        dimensions = input("Enter GridWorld dimensions as [len_x, len_y]: ").strip()
        try:
            dimensions = dimensions.strip("[]")
            dimensions = [int(x.strip()) for x in dimensions.split(",")]
            if len(dimensions) == 2 and all(x > 0 for x in dimensions):
                dimension_type_validity = True
            else:
                print("Ensure both dimensions are positive integers.")
        except ValueError:
            print("Invalid input. Enter dimensions as [x, y] with positive integers.")

    len_x, len_y = dimensions
    print(f"Gridworld dimensions set to: {len_x} x {len_y}")

    # Get reward cells (list of tuples)
    reward_cells_validity = False
    while not reward_cells_validity:
        print("\n")
        reward_cells = input(f"Enter reward cell coordinates as [(x1, y1), (x2, y2), ...]. Grid size is {len_x}x{len_y}. Use zero indexing: ").strip()
        try:
            reward_cells = eval(reward_cells)  # Convert to list of tuples

            if (
                isinstance(reward_cells, list)
                and all(isinstance(t, tuple) and len(t) == 2 for t in reward_cells)
                and all(0 <= t[0] < len_x and 0 <= t[1] < len_y for t in reward_cells)  # ✅ Check bounds
            ):
                reward_cells_validity = True
            else:
                print(f"Invalid coordinates. Ensure tuples are within (0 ≤ x < {len_x}, 0 ≤ y < {len_y}).")
        
        except (SyntaxError, ValueError, TypeError):
            print("Invalid input. Enter coordinates as [(x1, y1), (x2, y2), ...].")

    print(f"Reward cells set to: {reward_cells}")

    # Get reward values (list of numbers, same length as reward cells)
    reward_values_validity = False
    while not reward_values_validity:
        reward_values = input(f"Enter reward values as a list [{len(reward_cells)} values]: ").strip()
        try:
            reward_values = eval(reward_values)  # Convert to list
            if isinstance(reward_values, list) and len(reward_values) == len(reward_cells) and all(isinstance(x, (int, float)) for x in reward_values):
                reward_values_validity = True
            else:
                print(f"Invalid format. Enter {len(reward_cells)} numbers in a list, e.g., [5, -10, 3].")
        except (SyntaxError, ValueError, TypeError):
            print(f"Invalid input. Enter a list of {len(reward_cells)} numeric values.")

    print(f"Reward values set to: {reward_values}")

    # Get edge penalty (negative number)
    edge_penalty_validity = False
    while not edge_penalty_validity:
        edge_penalty = input("Enter the edge penalty (negative number): ").strip()
        try:
            edge_penalty = float(edge_penalty)
            if edge_penalty < 0:
                edge_penalty_validity = True
            else:
                print("Edge penalty must be a negative number.")
        except ValueError:
            print("Invalid input. Enter a negative number.")

    print(f"Edge penalty set to: {edge_penalty}")

    # Get probability of misstep (0 < p < 1)
    probability_of_misstep_validity = False
    while not probability_of_misstep_validity:
        probability_of_misstep = input("Enter probability of misstep (0 < p < 1): ").strip()
        try:
            probability_of_misstep = float(probability_of_misstep)
            if 0 < probability_of_misstep < 1:
                probability_of_misstep_validity = True
            else:
                print("Probability must be greater than 0 and less than 1.")
        except ValueError:
            print("Invalid input. Enter a number between 0 and 1.")

    print(f"Probability of misstep set to: {probability_of_misstep}")
    # ✅ Get max iterations (positive integer)
    max_iterations_validity = False
    while not max_iterations_validity:
        max_iterations = input("Enter the maximum number of iterations (positive integer): ").strip()
        try:
            max_iterations = int(max_iterations)
            if max_iterations > 0:
                max_iterations_validity = True
            else:
                print("Must be a positive integer.")
        except ValueError:
            print("Invalid input. Enter a positive integer.")

    # Get tolerance (small positive float)
    tolerance_validity = False
    while not tolerance_validity:
        tolerance = input("Enter the solver tolerance (e.g., 1e-6): ").strip()
        try:
            tolerance = float(tolerance)
            if tolerance > 0:
                tolerance_validity = True
            else:
                print("Tolerance must be a positive number.")
        except ValueError:
            print("Invalid input. Enter a small positive float (e.g., 0.00001 or 1e-5).")

    # Get discount rate
    discount_rate_validity = False
    while not discount_rate_validity:
        discount_rate = input("Enter the discount rate (0 ≤ γ < 1): ").strip()
        try:
            discount_rate = float(discount_rate)
            if 0 <= discount_rate < 1:
                discount_rate_validity = True
            else:
                print("Discount rate must be in the range [0, 1).")
        except ValueError:
            print("Invalid input. Enter a float between 0 and 1 (e.g., 0.9).")

    # Create GridWorld instance
    gridworld = Gridworld_Constructor(
        reward_states=reward_cells, 
        reward_values=reward_values, 
        probability_of_intended_move = 1 - probability_of_misstep, 
        len_x=len_x, 
        len_y=len_y, 
        border_penalty=edge_penalty
    ) 
    probabilities, rewards = gridworld()

    print("\nGridWorld successfully created! Initialising solver")

    solver = GenericMDP(states = [(i, j) for i in range(len_x) for j in range(len_y)], 
                        actions = [(1, 0),    # right
                                   (-1, 0),   # left
                                   (0, 1),    # up
                                   (0, -1)], 
                        probabilities = probabilities, 
                        rewards = rewards, 
                        discount_rate = discount_rate, 
                        max_iterations = max_iterations, 
                        len_x = dimensions[0], 
                        len_y = dimensions[1], 
                        reward_list = reward_cells, 
                        reward_values = reward_values, 
                        problem_type = 'gridworld')


if __name__ == "__main__":
    Value_Iteration()

           

