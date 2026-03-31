import sys

def rescale_ply(filepath, scale=1000):
    """Rescale vertex positions in an ASCII PLY file by the given factor."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find end of header and vertex count
    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        if line.strip() == 'end_header':
            header_end = i + 1
            break

    # Rescale x, y, z of each vertex line
    new_lines = lines[:header_end]
    for i in range(header_end, header_end + num_vertices):
        parts = lines[i].split()
        x = float(parts[0]) * scale
        y = float(parts[1]) * scale
        z = float(parts[2]) * scale
        rest = ' '.join(parts[3:])
        new_lines.append(f'{x} {y} {z} {rest}\n')

    # Keep face lines unchanged
    for i in range(header_end + num_vertices, len(lines)):
        new_lines.append(lines[i])

    with open(filepath, 'w') as f:
        f.writelines(new_lines)

    print(f'Rescaled {num_vertices} vertices by {scale}x in {filepath}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 rescale_mesh.py <file.ply>')
        sys.exit(1)
    rescale_ply(sys.argv[1])