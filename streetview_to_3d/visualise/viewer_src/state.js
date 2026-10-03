// Interaction state, independent of scene geometry and the DOM.
// Mode changes never clear selection, visibility or edit history.
export class ViewerState {
  mode = 'inspect';
  gun = false; // Fly with the gun (Shoot) instead of the bird
  selected = null;
  selectionKind = 'piece';
  hidden = new Set();
  groups = [];
  select(members, kind = 'piece') {
    this.selectionKind = kind;
    this.selected = members;
    if (members) members.forEach((i) => this.hidden.delete(i));
  }
  regroup(groups) {
    const anchor = this.selected?.[0];
    this.groups = groups;
    this.selected =
      anchor == null
        ? null
        : this.selectionKind === 'node'
          ? [anchor]
          : groups.find((m) => m.includes(anchor)) || null;
  }
  visible(i) {
    return !this.hidden.has(i);
  }
  toggleVisibility(members) {
    const visible = members.some((i) => this.visible(i));
    members.forEach((i) => (visible ? this.hidden.add(i) : this.hidden.delete(i)));
    if (visible && this.selected?.some((i) => members.includes(i))) this.select(null);
  }
  reset() {
    this.mode = 'inspect';
    this.selected = null;
    this.selectionKind = 'piece';
    this.hidden.clear();
    this.groups = [];
  }
}
