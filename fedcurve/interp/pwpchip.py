import numpy as np
from scipy.interpolate import PchipInterpolator
from collections import defaultdict


def create_piecewise_pchip(x_data, y_data):
    # Convert inputs to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Group data by x values
    x_groups = defaultdict(list)
    for x, y in zip(x_data, y_data):
        x_groups[x].append(y)

    # Sort the x values and determine boundaries
    unique_x = sorted(x_groups.keys())
    boundaries = []

    for x in unique_x:
        if len(x_groups[x]) > 1:
            boundaries.append(x)

    # Create interpolators for each segment
    interpolators = []

    if not boundaries:
        # No duplicates, just use standard PCHIP
        interpolators.append(
            {
                "min_x": -np.inf,
                "max_x": np.inf,
                "interpolator": PchipInterpolator(x_data, y_data),
            }
        )
    else:
        # Create segment data
        segments = []

        # First segment (before first boundary)
        first_segment_x = []
        first_segment_y = []

        for x in unique_x:
            if x < boundaries[0]:
                first_segment_x.append(x)
                first_segment_y.append(x_groups[x][0])  # Only one y value per x here
            elif x == boundaries[0]:
                first_segment_x.append(x)
                first_segment_y.append(max(x_groups[x]))  # Max y for first occurrence

        if len(first_segment_x) >= 2:  # Need at least 2 points for interpolation
            segments.append(
                {
                    "min_x": -np.inf,
                    "max_x": boundaries[0],
                    "x": first_segment_x,
                    "y": first_segment_y,
                }
            )

        # Middle segments (between boundaries)
        for i in range(len(boundaries) - 1):
            current_boundary = boundaries[i]
            next_boundary = boundaries[i + 1]

            segment_x = []
            segment_y = []

            # Add min y from current boundary
            segment_x.append(current_boundary)
            segment_y.append(min(x_groups[current_boundary]))

            # Add points between boundaries
            for x in unique_x:
                if current_boundary < x < next_boundary:
                    segment_x.append(x)
                    segment_y.append(x_groups[x][0])  # Only one y per x here

            # Add max y from next boundary
            segment_x.append(next_boundary)
            segment_y.append(max(x_groups[next_boundary]))

            if len(segment_x) >= 2:  # Need at least 2 points for interpolation
                segments.append(
                    {
                        "min_x": current_boundary,
                        "max_x": next_boundary,
                        "x": segment_x,
                        "y": segment_y,
                    }
                )

        # Last segment (after last boundary)
        last_segment_x = []
        last_segment_y = []

        # Add min y from last boundary
        last_segment_x.append(boundaries[-1])
        last_segment_y.append(min(x_groups[boundaries[-1]]))

        # Add remaining points
        for x in unique_x:
            if x > boundaries[-1]:
                last_segment_x.append(x)
                last_segment_y.append(x_groups[x][0])  # Only one y per x here

        if len(last_segment_x) >= 2:  # Need at least 2 points for interpolation
            segments.append(
                {
                    "min_x": boundaries[-1],
                    "max_x": np.inf,
                    "x": last_segment_x,
                    "y": last_segment_y,
                }
            )

        # Create interpolators for each segment
        for segment in segments:
            interpolators.append(
                {
                    "min_x": segment["min_x"],
                    "max_x": segment["max_x"],
                    "interpolator": PchipInterpolator(segment["x"], segment["y"]),
                }
            )

    # Store duplicate x values and their corresponding y values for exact matches
    duplicate_values = {x: x_groups[x] for x in x_groups if len(x_groups[x]) > 1}

    # Define the interpolation function
    def interpolate_func(x):
        """
        Evaluate the piecewise PCHIP interpolation at the given points.

        Parameters:
        -----------
        x : float or array-like
            Points at which to evaluate the interpolation

        Returns:
        --------
        y : float or array
            Interpolated values at x
        """
        x_array = np.atleast_1d(x)
        y_result = np.zeros_like(x_array, dtype=float)

        for i, x_val in enumerate(x_array):
            # Check if x is exactly a duplicate value
            if x_val in duplicate_values:
                # Duplicate x value - use min or max depending on which side we're approaching from
                y_result[i] = min(
                    duplicate_values[x_val]
                )  # Default to min (for x >= boundary)
            else:
                # Find the appropriate interpolator based on which segment contains x
                for interp in interpolators:
                    if interp["min_x"] <= x_val < interp["max_x"] or (
                        x_val == interp["max_x"] and interp["max_x"] != np.inf
                    ):
                        y_result[i] = interp["interpolator"](x_val)
                        break

        # Return scalar if input was scalar
        if np.isscalar(x):
            return y_result[0]
        return y_result

    return interpolate_func


# Example usage
if __name__ == "__main__":
    # Example data with duplicate x values
    x = [1, 2, 3, 4, 4, 4, 5, 6, 8, 8, 10]
    y = [10, 8, 6, 5, 4, 3, 2.5, 2, 1.5, 1, 0.5]

    # Create interpolation function
    interp_func = create_piecewise_pchip(x, y)

    # Example: evaluate at specific points
    x_test = [1.5, 3.5, 4.0, 7.0, 8.0, 9.0]
    for xt in x_test:
        print(f"f({xt}) = {interp_func(xt)}")

    # Example: plot the function
    import matplotlib.pyplot as plt

    # Query points for plotting
    x_query = np.linspace(0, 11, 1000)
    y_interp = interp_func(x_query)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o", label="Data points")
    plt.plot(x_query, y_interp, "-", label="Piecewise PCHIP")

    # Highlight duplicate x values
    for x_val in set([x_val for x_val in x if x.count(x_val) > 1]):
        plt.axvline(x=x_val, color="r", linestyle="--", alpha=0.3)

    plt.legend()
    plt.grid(True)
    plt.title("Piecewise PCHIP Interpolation with Duplicate X Values")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
