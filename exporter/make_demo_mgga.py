"""Create the untrained meta-GGA demo checkpoints (MGGA_F{x,c}_demo) used to validate
the tau path of the exporter.  Random equinox initialisation, fixed seeds."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exporter import eqx_classes  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eqx_demo')

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fx = eqx_classes.MGGA_Fx_demo(depth=3, nodes=16, seed=7)
    fc = eqx_classes.MGGA_Fc_demo(depth=3, nodes=16, seed=11)
    for m in (fx, fc):
        eqx_classes.save_model(m, os.path.join(OUT, f'{m.name}_d3_n16_s{m.seed}'))
    print('wrote', sorted(os.listdir(OUT)))
