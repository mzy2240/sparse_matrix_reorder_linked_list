import numpy as np
from typing import Union
from itertools import combinations
from copy import deepcopy


class Node:
    def __init__(self, data=None, nref=None, pref=None):
        self.value = data
        self.nref = nref
        self.pref = pref

    def __eq__(self, other):
        return self.value == other.value


class SparseMatrixElement(Node):

    def __init__(self, col, *args):
        super().__init__(*args)
        self.col = col

    def __repr__(self):
        return f"SparseMatrixElementObject Col: {self.col} Value: {self.value}"

    def __eq__(self, other):
        return self.col == other.col and self.value == other.value


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.index = None

    def __repr__(self):
        if self.head is None:
            return "List has no element"
        else:
            list_string = []
            n = self.head
            print(self.head)
            while n is not None:
                list_string.append(f"{n.col}:{n.value}")
                print(list_string)
                n = n.nref
            return ','.join(list_string)

    @property
    def length(self):
        count = 0
        node = self.head
        while node:
            count += 1
            node = node.nref
        return count

    def insert_in_emptylist(self, data):
        if self.head is None:
            new_node = Node(data)
            self.head = new_node
        else:
            print("list is not empty")

    def insert_at_start(self, data):
        new_node = Node(data)
        new_node.nref = self.head
        if self.head is not None:
            self.head.pref = new_node
        print("node inserted")
        self.head = new_node

    def insert_at_end(self, new_node: SparseMatrixElement):
        new_node.nref = None
        if self.head is None:
            new_node.pref = None
            self.head = new_node
            return
        last = self.head
        while last.nref is not None:
            last = last.nref
        last.nref = new_node
        new_node.pref = last
        return

    def insert_after_item(self, prev_node, data):
        if self.head is None:
            print("List is empty")
            return
        new_node = Node(data, nref=prev_node.nref, pref=prev_node)
        prev_node.next = new_node
        if new_node.nref is not None:
            new_node.nref.pref = new_node

    def insert_before_item(self, x, data):
        if self.head is None:
            print("List is empty")
            return
        else:
            n = self.head
            while n is not None:
                if n.value == x:
                    break
                n = n.nref
            if n is None:
                print("item not in the list")
            else:
                new_node = Node(data)
                new_node.nref = n
                new_node.pref = n.pref
                if n.pref is not None:
                    n.pref.nref = new_node
                n.pref = new_node

    def insert_fill_at_col(self, col):
        new_node = SparseMatrixElement(col, 1)
        node = self.head
        if node.col > new_node.col:
            new_node.nref = self.head
            if self.head:
                self.head.pref = new_node
            self.head = new_node
        else:
            while node:
                if node.col > new_node.col:
                    new_node.nref = node
                    new_node.pref = node.pref
                    if node.pref is not None:
                        node.pref.nref = new_node
                    node.pref = new_node
                    return
                node = node.nref
            self.insert_at_end(new_node)

    def traverse_list(self):
        if self.head is None:
            print("List has no element")
            return
        else:
            n = self.head
            while n is not None:
                print(n.value, " ")
                n = n.nref

    def delete_at_start(self):
        if self.head is None:
            print("The list has no element to delete")
            return
        if self.head.nref is None:
            self.head = None
            return
        self.head = self.head.nref
        self.start_prev = None

    def delete_at_end(self):
        if self.head is None:
            print("The list has no element to delete")
            return
        if self.head.nref is None:
            self.head = None
            return
        n = self.head
        while n.nref is not None:
            n = n.nref
        n.pref.nref = None

    def delete_element_by_value(self, x):
        if self.head is None:
            print("The list has no element to delete")
            return
        if self.head.nref is None:
            if self.head.value == x:
                self.head = None
            else:
                print("Item not found")
            return

        if self.head.value == x:
            self.head = self.head.nref
            self.head.pref = None
            return

        n = self.head
        while n.nref is not None:
            if n.value == x:
                break
            n = n.nref
        if n.nref is not None:
            n.pref.nref = n.nref
            n.nref.pref = n.pref
        else:
            if n.value == x:
                n.pref.nref = None
            else:
                print("Element not found")

    def delete_element_by_col(self, col: int):
        node = self.head
        if node.col == col:
            self.head = node.nref
            self.head.pref = None
            return
        else:
            while node:
                if node.col == col:
                    if node.nref:
                        node.pref.nref = node.nref
                        node.nref.pref = node.pref
                    else:
                        node.pref.nref = None
                    break
                node = node.nref

    def reverse_linked_list(self):
        if self.head is None:
            print("The list has no element to delete")
            return
        p = self.head
        q = p.nref
        p.nref = None
        p.pref = q
        while q is not None:
            q.pref = q.nref
            q.nref = p
            p = q
            q = q.pref
        self.head = p

    def get_node(self, index):
        current = self.head
        count = 0
        while current:
            if count == index:
                return current
            count += 1
            current = current.nref
        assert False

    def get_node_by_col(self, col):
        current = self.head
        while current:
            if current.col == col:
                return current
            current = current.nref
        assert False

    def connected_to_node(self, col):
        current = self.head
        while current:
            if current.col == col:
                return True
            current = current.nref
        return False

    def permutate(self, p_vector):
        node = self.head
        list_col = []
        list_col_new = []
        list_value = []
        while node:
            list_col.append(node.col)
            list_value.append(node.value)
            node = node.nref
        for col in list_col:
            list_col_new.append(p_vector.index(col))
        list_pair = zip(list_value, list_col_new)
        list_pair_new = sorted(list_pair, key = lambda x: x[1])
        node = self.head
        for pair in list_pair_new:
            node.value, node.col = pair
            node = node.nref


def generate_sparse_matrix_from_raw(raw_matrix, ignore_zero: bool = True,
                                    min_value: Union[float, int] = 0):
    n_rows, n_cols = raw_matrix.shape
    row_head = []
    row_diag = []
    row_sparse_matrix = SparseMatrix()
    for i in range(n_rows):
        sr = DoublyLinkedList()  # sparse row
        for j in range(n_cols):
            if raw_matrix[i, j] != 0:
                sme = SparseMatrixElement(j, raw_matrix[i, j])
                sr.insert_at_end(sme)
        row_diag.append(sr.get_node_by_col(i))
        row_head.append(sr.head)
        row_sparse_matrix.append(sr)
    return row_head, row_diag, row_sparse_matrix


class SparseMatrix(list):

    def __repr__(self):
        list_str = []
        for index, row in enumerate(self):
            node = row.head
            while node:
                list_str.append(f"row {index} col {node.col} value "
                                f"{node.value}")
                node = node.nref
        return '\n'.join(list_str)

    def print_all(self):
        for index, row in enumerate(self):
            node = row.head
            while node:
                print(f"Row: {index}", node)
                node = node.nref

    def append(self, dl):
        dl.index = len(self)
        super(SparseMatrix, self).append(dl)

    def row_full(self, index):
        return load_sparse_working_row(self[index], len(self))

    def permutate(self, p_vector):
        for row in self:
            row.permutate(p_vector)
        self.__init__([self[i] for i in p_vector])


def load_sparse_working_row(dl: DoublyLinkedList, length: int):
    """
    It is actually the working row full vector.
    """
    node = dl.head
    swr = [0] * length
    while node:
        swr[node.col] = node.value
        node = node.nref
    return swr


def unload_sparse_working_row(dl: DoublyLinkedList, swr: list):
    node = dl.head
    while node:
        node.value = swr[node.col]
        swr[node.col] = 0
        node = node.nref
    return dl


def sparse_factorization(sparse_matrix, row_diag):
    n_dim = len(sparse_matrix)
    for i in range(1, n_dim):
        swr = load_sparse_working_row(sparse_matrix[i], n_dim)
        p2 = sparse_matrix[i].head
        while p2 != row_diag[i]:
            p1 = row_diag[p2.col]
            swr[p2.col] /= p1.value
            p1 = p1.nref
            while p1:
                swr[p1.col] -= swr[p2.col] * p1.value
                p1 = p1.nref
            p2 = p2.nref
        sparse_matrix[i] = unload_sparse_working_row(sparse_matrix[i], swr)
    return sparse_matrix


def sparse_factorization_using_permutation(sparse_matrix, row_diag, p_vector):
    n_dim = len(sparse_matrix)
    for i in range(1, n_dim):
        k = p_vector[i]
        swr = load_sparse_working_row(sparse_matrix[k], n_dim)
        p2 = sparse_matrix[k].head
        while p2 != row_diag[k]:
            p1 = row_diag[p2.col]
            swr[p2.col] /= p1.value
            p1 = p1.nref
            while p1:
                swr[p1.col] -= swr[p2.col] * p1.value
                p1 = p1.nref
            p2 = p2.nref
        sparse_matrix[k] = unload_sparse_working_row(sparse_matrix[k], swr)
    return sparse_matrix


def sparse_forward_substitution(sparse_matrix, row_diag, p_vector, b_vector):
    n_dim = len(sparse_matrix)
    for i in range(1, n_dim):
        k = p_vector[i]
        p1 = sparse_matrix[k].head
        while p1 != row_diag[k]:
            b_vector[k] -= p1.value*b_vector[p1.col]
            p1 = p1.nref
    return b_vector


def sparse_backward_substitution(sparse_matrix, row_diag, p_vector, b_vector):
    n_dim = len(sparse_matrix)
    for i in range(n_dim, 1, -1):
        k = p_vector[i]
        p1 = row_diag[k].nref
        while p1:
            b_vector[k] -= p1.value * b_vector[p1.col]
            p1 = p1.nref
        b_vector[k] /= row_diag[k].value
    return b_vector


def factorization_path(row_diag, arow):
    p1 = row_diag[arow]
    path = []
    while p1:
        path.append(p1.col)
        p1 = row_diag[p1.col].nref
    return path


def reorder_matrix(matrix, method='tinney2'):
    n_dim = len(matrix)
    chosen_nodes = [False] * n_dim
    bswr = [False] * n_dim
    row_perm = [0] * n_dim
    sorted_matrix = deepcopy(matrix)
    num_fills = 0

    for i in range(0, n_dim):
        sorted_matrix = sorted(sorted_matrix, key=lambda x: x.length)
        sorted_index = [x.index for x in sorted_matrix]
        k = sorted_matrix[0].index
        row_perm[i] = k
        chosen_nodes[k] = True
        adj_nodes = []
        node = sorted_matrix[0].head
        while node:
            if node.col != k:
                adj_nodes.append(node)
            node = node.nref

        if len(adj_nodes) > 1:
            pairs = combinations(adj_nodes, 2)
            for pair in pairs:
                index_1 = pair[0].col
                index_2 = pair[1].col
                if matrix[index_1].connected_to_node(index_2):
                    continue
                else:
                    matrix[index_1].insert_fill_at_col(index_2)
                    matrix[index_2].insert_fill_at_col(index_1)
                    index = sorted_index.index(index_1)
                    sorted_matrix[index].insert_fill_at_col(index_2)
                    index = sorted_index.index(index_2)
                    sorted_matrix[index].insert_fill_at_col(index_1)
                    num_fills += 2

        for node in adj_nodes:
            index = sorted_index.index(node.col)
            sorted_matrix[index].delete_element_by_col(k)
        sorted_matrix.pop(0)
    return row_perm, num_fills, matrix


if __name__ == "__main__":
    a = np.array(
        [[5, 0, 0, -4], [0, 4, 0, -3], [0, 0, 3, -2], [-4, -3, -2, 10]])
    b = np.array(
        [[10, -4, -3, -2], [-4, 5, 0, 0], [-3, 0, 4, 0], [-2, 0, 0, 3]])
    c = np.array([[1,1,0,1,1], [1,1,1,0,0], [0,1,1,1,0], [1,0,1,1,1], [1,0,
                                                                       0,1,1]])
    spm = []
    hp, dp, lil = generate_sparse_matrix_from_raw(b)
    permutation, num_fill, matrix = reorder_matrix(lil)
    print(permutation)
    print("--------------------")
    print(matrix)
    print("--------------------")
    print(num_fill)
    print("--------------------")
    matrix.permutate(permutation)
    print(matrix)

