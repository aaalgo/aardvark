#!/usr/bin/env python3
import os
from jinja2 import Environment, FileSystemLoader

TMPL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                './templates')
env = Environment(loader=FileSystemLoader(searchpath=TMPL_DIR))
tmpl = env.get_template('gallery.html')

class Gallery:
    def __init__ (self, path, cols = 1, header = None, ext = '.png'):
        self.next_id = 0
        self.path = path
        self.cols = cols
        self.header = header
        self.ext = ext
        self.images = []
        try:
            if path != '.':
                os.makedirs(path)
        except:
            pass
        pass

    def text (self, tt, br = False):
        self.images.append({
            'text': tt})
        if br:
            for i in range(1, self.cols):
                self.images.append({
                    'text': ''})
        pass

    def next (self, text=None, link=None, ext=None, path=None):
        if ext is None:
            ext = self.ext
        if path is None:
            path = '%03d%s' % (self.next_id, ext)
        self.images.append({
            'image': path,
            'text': text,
            'link': link})
        self.next_id += 1
        return os.path.join(self.path, path)

    def flush (self):
        with open(os.path.join(self.path, 'index.html'), 'w') as f:
            images = [self.images[i:i+self.cols] for i in range(0, len(self.images), self.cols)]
            f.write(tmpl.render(images=images, header=self.header))
            pass
        pass

if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--ext", default='.jpg')
    args = parser.parse_args()
    gal = Gallery('.')
    for path in glob('*' + args.ext):
        print(path)
        gal.next(path=path)
        pass
    gal.flush()


