# PHANTOM: phageome-based disease classification, from viral metagenomic
# assemblies to case/control classifiers.
# Copyright (C) 2026  Radosław Pławecki, Kamil Szyc, Sophia Bałdysz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program (the file LICENSE).  If not, see
# <https://www.gnu.org/licenses/>.
#
# Contact: https://github.com/RadoslawPlawecki/PHANTOM/issues
"""``python -m phantom`` -- same as the ``phantom`` command."""

from phantom.runner.cli import main

raise SystemExit(main())
