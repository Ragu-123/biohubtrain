from pathlib import Path

for root_name in ('/kaggle/input', '/kaggle/working'):
    root = Path(root_name)
    print('ROOT', root_name)
    for path in sorted(root.iterdir()):
        print(str(path))
        if path.is_dir():
            for child in sorted(path.iterdir())[:30]:
                print(' ', str(child))

for root_name in ('/kaggle/input/competitions/biohub-cell-tracking-during-development', '/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/src/biohub_tracking', '/kaggle/working/biohubtrain'):
    root = Path(root_name)
    print('EXPECTED_PATH', root_name, 'EXISTS', root.exists())
    if root.is_dir():
        for path in sorted(root.iterdir())[:30]:
            print(' ', str(path))
            if path.name == 'train' and path.is_dir():
                for child in sorted(path.iterdir())[:12]:
                    print('   ', str(child))
