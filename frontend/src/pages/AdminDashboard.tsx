import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAppStore } from '../store/appStore';
import { Shield, Plus, LogOut, Trash2, User, TrendingUp, Edit2, X, ShieldCheck, Eye, EyeOff } from 'lucide-react';

export default function AdminDashboard() {
  const [searchParams] = useSearchParams();
  const tabParam = searchParams.get('tab') as 'super_admin' | 'admin' | 'user' | 'profile' | 'roles' | null;
  const [activeTab, setActiveTab] = useState<'super_admin' | 'admin' | 'user' | 'profile' | 'roles'>(tabParam || 'user');
  const [accounts, setAccounts] = useState<any[]>([]);

  const [customRoles, setCustomRoles] = useState<any[]>([]);
  const [isRoleModalOpen, setIsRoleModalOpen] = useState(false);
  const [roleMode, setRoleMode] = useState<'create' | 'edit'>('create');
  const [roleName, setRoleName] = useState('');
  const [editingOldRoleName, setEditingOldRoleName] = useState('');


  // Update activeTab when query param changes
  useEffect(() => {
    if (tabParam) {
      setActiveTab(tabParam);
    }
  }, [tabParam]);

  // Modal states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');

  // Form states
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const SIDEBAR_OPTIONS = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'data', label: 'Data Manager' },
    { id: 'stocks', label: 'Stock Selector' },
    { id: 'regime', label: 'Regime Analysis' },
    { id: 'models', label: 'Model Studio' },
    { id: 'patterns', label: 'Pattern Lab' },
    { id: 'backtest', label: 'Backtest Engine' },
    { id: 'trading', label: 'Live Trading' },
    { id: 'portfolio', label: 'Portfolio' },
    { id: 'settings', label: 'Settings' },
    { id: 'theme', label: 'Appearance' },
    { id: 'create_user', label: 'Create User' },
    { id: 'create_super_admin', label: 'Create Super Admin' }
  ];

  const [role, setRole] = useState<string>('user');
  const [permissions, setPermissions] = useState<string[]>(SIDEBAR_OPTIONS.map(o => o.id));
  const [showPassword, setShowPassword] = useState(false);
  const [editingOldUsername, setEditingOldUsername] = useState('');

  // Profile states
  const [profileUsername, setProfileUsername] = useState('');
  const [profilePassword, setProfilePassword] = useState('');
  const [profileEmail, setProfileEmail] = useState('');
  const [profilePhone, setProfilePhone] = useState('');

  const currentUserRole = localStorage.getItem('aitrade-current-role');
  const currentUser = localStorage.getItem('aitrade-current-user');
  const navigate = useNavigate();
  const { addNotification } = useAppStore();

  // Populate profile info
  useEffect(() => {
    if (activeTab === 'profile') {
      const currentUser = localStorage.getItem('aitrade-current-user');
      const storedUsersStr = localStorage.getItem('aitrade-users');
      if (storedUsersStr && currentUser) {
        const users = JSON.parse(storedUsersStr);
        const me = users.find((u: any) => u.username === currentUser);
        if (me) {
          setProfileUsername(me.username);
          setProfilePassword(me.password || '');
          setProfileEmail(me.email || '');
          setProfilePhone(me.phone || '');
        }
      }
    }
  }, [activeTab, accounts]);

  useEffect(() => {
    const role = localStorage.getItem('aitrade-current-role');

    // User role can only view profile tab
    if (role === 'user' && tabParam !== 'profile') {
      navigate('/dashboard');
      return;
    }

    // Admins cannot view super_admin or admin tabs
    if (role === 'admin' && (tabParam === 'super_admin' || tabParam === 'admin')) {
      navigate('/admin?tab=user');
      return;
    }

    if (!role) {
      navigate('/');
      return;
    }


    // Admins cannot view super_admin or admin tabs
    if (role === 'admin' && (tabParam === 'super_admin' || tabParam === 'admin')) {
      navigate('/admin?tab=user');
      return;
    }

    loadData();
  }, [navigate, tabParam]);


  const loadData = () => {
    const storedUsersStr = localStorage.getItem('aitrade-users');
    if (storedUsersStr) {
      setAccounts(JSON.parse(storedUsersStr));
    } else {
      setAccounts([]);
    }
    const storedRolesStr = localStorage.getItem('aitrade-roles');
    if (storedRolesStr) {
      setCustomRoles(JSON.parse(storedRolesStr));
    } else {
      const initialRoles = [
        { name: 'super_admin', permissions: SIDEBAR_OPTIONS.map(o => o.id) },
        { name: 'admin', permissions: SIDEBAR_OPTIONS.map(o => o.id) },
        { name: 'user', permissions: ['dashboard', 'trading', 'portfolio'] }
      ];
      setCustomRoles(initialRoles);
      localStorage.setItem('aitrade-roles', JSON.stringify(initialRoles));
    }
  };

  const handleOpenCreateModal = (forceRole?: string) => {
    setModalMode('create');
    setUsername('');
    setPassword('');
    setEmail('');
    setPhone('');
    setRole(forceRole || (activeTab === 'admin' ? 'admin' : 'user'));
    setPermissions(SIDEBAR_OPTIONS.map(o => o.id)); // Default all checked
    setIsModalOpen(true);
  };


  const handleOpenRoleModal = (role?: any) => {
    if (role) {
      setRoleMode('edit');
      setEditingOldRoleName(role.name);
      setRoleName(role.name);
      setPermissions(role.permissions || []);
    } else {
      setRoleMode('create');
      setRoleName('');
      setPermissions(SIDEBAR_OPTIONS.map(o => o.id));
    }
    setIsRoleModalOpen(true);
  };

  const handleRoleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!roleName) {
      addNotification({ type: 'error', message: 'Please provide role name' });
      return;
    }
    let updatedRoles = [...customRoles];
    if (roleMode === 'create') {
      if (updatedRoles.find(r => r.name.toLowerCase() === roleName.toLowerCase())) {
        addNotification({ type: 'error', message: 'Role already exists' });
        return;
      }
      updatedRoles.push({ name: roleName, permissions });
      addNotification({ type: 'success', message: `Role ${roleName} created successfully!` });
    } else {
      updatedRoles = updatedRoles.map(r => r.name === editingOldRoleName ? { ...r, name: roleName, permissions } : r);
      addNotification({ type: 'success', message: `Role ${roleName} updated successfully!` });
    }
    setCustomRoles(updatedRoles);
    localStorage.setItem('aitrade-roles', JSON.stringify(updatedRoles));
    setIsRoleModalOpen(false);
  };

  const handleDeleteRole = (nameToDelete: string) => {
    if (window.confirm(`Are you sure you want to delete role ${nameToDelete}?`)) {
      const updatedRoles = customRoles.filter(r => r.name !== nameToDelete);
      setCustomRoles(updatedRoles);
      localStorage.setItem('aitrade-roles', JSON.stringify(updatedRoles));
      addNotification({ type: 'info', message: `Role ${nameToDelete} deleted` });
    }
  };

  const handleOpenEditModal = (account: any) => {
    setModalMode('edit');
    setEditingOldUsername(account.username);
    setUsername(account.username);
    setPassword(account.password || '');
    setEmail(account.email || '');
    setPhone(account.phone || '');
    setRole(account.role || 'user');
    setPermissions(account.permissions || SIDEBAR_OPTIONS.map(o => o.id));
    setIsModalOpen(true);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!username) { addNotification({ type: 'error', message: 'Please enter username' }); return; }
    if (!email) { addNotification({ type: 'error', message: 'Please enter email id' }); return; }
    if (!phone) { addNotification({ type: 'error', message: 'Please enter phone number' }); return; }
    if (phone.length !== 10) { addNotification({ type: 'error', message: 'Phone number must be exactly 10 digits' }); return; }
    if (!password) { addNotification({ type: 'error', message: 'Please enter password' }); return; }

    const storageKey = 'aitrade-users';
    const storedDataStr = localStorage.getItem(storageKey);
    let storedData = storedDataStr ? JSON.parse(storedDataStr) : [];

    if (modalMode === 'create') {
      if (storedData.find((u: any) => u.username === username)) {
        addNotification({ type: 'error', message: 'Account already exists' });
        return;
      }
      storedData.push({ username, password, email, phone, role, permissions, createdBy: currentUser });
      const displayRole = role === 'admin' ? 'Admin' : 'User';
      addNotification({ type: 'success', message: `${displayRole} ${username} created successfully!` });
    } else {
      // Edit mode
      const userExists = storedData.find((u: any) => u.username === username && u.username !== editingOldUsername);
      if (userExists) {
        addNotification({ type: 'error', message: 'Username is already taken' });
        return;
      }

      storedData = storedData.map((u: any) =>
        u.username === editingOldUsername ? { ...u, username, password, email, phone, role, permissions } : u
      );
      const displayRole = role === 'admin' ? 'Admin' : 'User';
      addNotification({ type: 'success', message: `${displayRole} ${username} updated successfully!` });
    }

    localStorage.setItem(storageKey, JSON.stringify(storedData));
    setIsModalOpen(false);
    loadData();
  };

  const handleProfileUpdate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!profileUsername) { addNotification({ type: 'error', message: 'Please enter username' }); return; }
    if (!profileEmail) { addNotification({ type: 'error', message: 'Please enter email id' }); return; }
    if (!profilePhone) { addNotification({ type: 'error', message: 'Please enter phone number' }); return; }
    if (profilePhone.length !== 10) { addNotification({ type: 'error', message: 'Phone number must be exactly 10 digits' }); return; }
    if (!profilePassword) { addNotification({ type: 'error', message: 'Please enter password' }); return; }

    const currentUser = localStorage.getItem('aitrade-current-user');
    const storageKey = 'aitrade-users';
    const storedDataStr = localStorage.getItem(storageKey);
    let storedData = storedDataStr ? JSON.parse(storedDataStr) : [];

    const meIndex = storedData.findIndex((u: any) => u.username === currentUser);
    if (meIndex === -1) return;

    // Check if new username is taken
    const taken = storedData.find((u: any) => u.username === profileUsername && u.username !== currentUser);
    if (taken) {
      addNotification({ type: 'error', message: 'Username is already taken' });
      return;
    }

    storedData[meIndex].username = profileUsername;
    storedData[meIndex].password = profilePassword;
    storedData[meIndex].email = profileEmail;
    storedData[meIndex].phone = profilePhone;

    localStorage.setItem(storageKey, JSON.stringify(storedData));
    localStorage.setItem('aitrade-current-user', profileUsername); // Update active session
    addNotification({ type: 'success', message: 'Profile updated successfully!' });
    loadData();
  };

  const handleDelete = (usernameToDelete: string) => {
    if (window.confirm(`Are you sure you want to delete ${usernameToDelete}?`)) {
      const storageKey = 'aitrade-users';
      const storedDataStr = localStorage.getItem(storageKey);
      let storedData = storedDataStr ? JSON.parse(storedDataStr) : [];

      const userToDelete = storedData.find((u: any) => u.username === usernameToDelete);
      storedData = storedData.filter((u: any) => u.username !== usernameToDelete);
      localStorage.setItem(storageKey, JSON.stringify(storedData));

      const displayRole = userToDelete?.role === 'admin' ? 'Admin' : 'User';
      addNotification({ type: 'info', message: `${displayRole} ${usernameToDelete} deleted` });
      loadData();
    }
  };

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="p-8 max-w-5xl mx-auto w-full">
        <div className="animate-[fade-in_0.2s_ease-out]">

          {/* Header with Title and Create Button */}
          {activeTab !== 'profile' ? (
            <div className="flex items-center justify-between mb-8">
              <div>
                <h1 className="text-2xl font-bold text-[var(--text)] mb-1 capitalize">
                  {activeTab.replace('_', ' ')} Management
                </h1>
                <p className="text-[var(--text-muted)]">
                  Manage system access for {activeTab.replace('_', ' ')}s.
                </p>
              </div>

              <button
                onClick={() => activeTab === 'roles' ? handleOpenRoleModal() : handleOpenCreateModal()}

                className="flex items-center gap-2 bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white px-4 py-2 rounded-lg font-medium transition-colors shadow-sm"
              >
                <Plus size={18} />
                Create New {activeTab === 'roles' ? 'Role' : activeTab === 'super_admin' ? 'Super Admin' : activeTab === 'admin' ? 'Admin' : 'User'}
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-between mb-8">
              <div>
                <h1 className="text-3xl font-bold text-[var(--text)] mb-2 flex items-center gap-3">
                  <User className="text-[var(--primary)]" size={28} />
                  My Profile
                </h1>
                <p className="text-[var(--text-muted)]">
                  Manage your personal account credentials.
                </p>
              </div>
            </div>
          )}


          {/* Roles Management View */}
          {activeTab === 'roles' && (
            <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl shadow-sm overflow-hidden mt-4">
              <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg)]">
                <h3 className="font-semibold text-[var(--text)] flex items-center gap-2">
                  <ShieldCheck size={16} className="text-purple-400" />
                  {customRoles.length} Role(s) Defined
                </h3>
              </div>
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[var(--bg)] border-b border-[var(--border)]">
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider w-20">S.No</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Role Name</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Access Modules</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider text-right w-32">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {customRoles.map((roleObj, index) => (
                    <tr key={roleObj.name} className="hover:bg-[var(--bg-hover)] transition-colors">
                      <td className="px-6 py-4 text-[var(--text-muted)] font-medium">{index + 1}</td>
                      <td className="px-6 py-4 text-[var(--text)] font-medium capitalize">{roleObj.name.replace('_', ' ')}</td>
                      <td className="px-6 py-4">
                        <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${(roleObj.permissions || []).length === SIDEBAR_OPTIONS.length
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : 'bg-blue-500/10 text-blue-400'
                          }`}>
                          {(roleObj.permissions || []).length} modules
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center justify-end gap-2">
                          <button onClick={() => handleOpenRoleModal(roleObj)} className="p-2 text-[var(--text-muted)] hover:text-blue-400 hover:bg-blue-400/10 rounded-md transition-colors" title="Edit Role">
                            <Edit2 size={16} />
                          </button>
                          <button onClick={() => handleDeleteRole(roleObj.name)} className="p-2 text-[var(--text-muted)] hover:text-rose-400 hover:bg-rose-400/10 rounded-md transition-colors" title="Delete Role">
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}



          {/* User Profile View */}
          {activeTab === 'profile' && (
            <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl shadow-sm p-8 max-w-md mt-4">
              <div className="w-16 h-16 rounded-full bg-[var(--primary-subtle)] flex items-center justify-center mb-6 shadow-sm border border-[var(--primary-glow)]">
                <User size={32} className="text-[var(--primary)]" />
              </div>
              <form onSubmit={handleProfileUpdate} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Username</label>
                  <input
                    type="text"
                    value={profileUsername}
                    onChange={(e) => setProfileUsername(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                    placeholder="Enter username"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Email</label>
                  <input
                    type="email"
                    value={profileEmail}
                    onChange={(e) => setProfileEmail(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                    placeholder="Enter email id"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Phone Number</label>
                  <input
                    type="text"
                    value={profilePhone}
                    onChange={(e) => setProfilePhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                    pattern="[0-9]{10}"
                    maxLength={10}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                    placeholder="Enter 10-digit phone number"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Password</label>
                  <div className="relative">
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={profilePassword}
                      onChange={(e) => setProfilePassword(e.target.value)}
                      className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors pr-10"
                      placeholder="Enter password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
                    >
                      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
                <div className="pt-4">
                  <button
                    type="submit"
                    className="w-full bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white px-4 py-2.5 rounded-lg font-medium transition-colors shadow-sm"
                  >
                    Save Changes
                  </button>
                </div>
              </form>
            </div>
          )}

          {/* View for activeTab === 'admin' or 'user' */}
          {activeTab !== 'super_admin' && activeTab !== 'roles' && activeTab !== 'profile' && (
            <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl shadow-sm overflow-hidden mt-4">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[var(--bg)] border-b border-[var(--border)]">
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider w-20">S.No</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Username</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Email</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Phone</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Role</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Access</th>
                    <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider text-right w-32">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {accounts.filter(a => currentUserRole === 'admin' ? (a.role !== 'super_admin' && a.role !== 'admin' && a.createdBy === currentUser) : true).length === 0 ? (
                    <tr>
                      <td colSpan={8} className="px-6 py-8 text-center text-[var(--text-muted)]">
                        No user accounts created yet. Click "Create New" to add one.
                      </td>
                    </tr>
                  ) : (
                    accounts.filter(a => currentUserRole === 'admin' ? (a.role !== 'super_admin' && a.role !== 'admin' && a.createdBy === currentUser) : true).map((account, index) => (
                      <tr key={account.username} className="hover:bg-[var(--bg-hover)] transition-colors">
                        <td className="px-6 py-4 text-[var(--text-muted)] font-medium">
                          {index + 1}
                        </td>
                        <td className="px-6 py-4 text-[var(--text)] font-medium">
                          {account.username}
                        </td>
                        <td className="px-6 py-4 text-[var(--text-muted)] text-sm">
                          {account.email || '-'}
                        </td>
                        <td className="px-6 py-4 text-[var(--text-muted)] text-sm">
                          {account.phone || '-'}
                        </td>
                        <td className="px-6 py-4">
                          <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${account.role === 'super_admin' ? 'bg-amber-500/10 text-amber-400' :
                            account.role === 'admin' ? 'bg-purple-500/10 text-purple-400' : 'bg-blue-500/10 text-blue-400'
                            }`}>
                            {account.role?.replace('_', ' ') || 'user'}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${(account.permissions || []).length === SIDEBAR_OPTIONS.length
                            ? 'bg-emerald-500/10 text-emerald-400'
                            : 'bg-blue-500/10 text-blue-400'
                            }`}>
                            {(account.permissions || []).length} modules
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={() => handleOpenEditModal(account)}
                              className="p-2 text-[var(--text-muted)] hover:text-blue-400 hover:bg-blue-400/10 rounded-md transition-colors"
                              title="Edit user"
                            >
                              <Edit2 size={16} />
                            </button>
                            <button
                              onClick={() => handleDelete(account.username)}
                              className="p-2 text-[var(--text-muted)] hover:text-rose-400 hover:bg-rose-400/10 rounded-md transition-colors"
                              title="Delete"
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>


      {/* Modal Overlay for Create/Edit Role */}
      {isRoleModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-[fade-in_0.15s_ease-out]">
          <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl shadow-2xl w-full max-w-md overflow-hidden transform scale-100 transition-all">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] bg-[var(--bg)]">
              <h3 className="text-lg font-semibold text-[var(--text)]">
                {roleMode === 'create' ? 'Create New Role' : 'Edit Role'}
              </h3>
              <button onClick={() => setIsRoleModalOpen(false)} className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)] rounded-md transition-colors">
                <X size={18} />
              </button>
            </div>
            <form onSubmit={handleRoleSubmit} className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Role Name</label>
                <input
                  type="text"
                  value={roleName}
                  onChange={(e) => setRoleName(e.target.value)}
                  className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                  placeholder="e.g. manager, analyst"
                  disabled={false}
                />
              </div>
              <div className="pt-2">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-[var(--text-secondary)]">Sidebar Access</label>
                  <button type="button" onClick={() => setPermissions(permissions.length === SIDEBAR_OPTIONS.length ? [] : SIDEBAR_OPTIONS.map(o => o.id))} className="text-xs text-[var(--primary)] hover:underline">
                    {permissions.length === SIDEBAR_OPTIONS.length ? 'Deselect All' : 'Select All'}
                  </button>
                </div>
                <div className="bg-[var(--bg)] border border-[var(--border)] rounded-lg p-3 grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                  {SIDEBAR_OPTIONS.map(option => (
                    <label key={option.id} className="flex items-center gap-2 cursor-pointer group">
                      <input
                        type="checkbox"
                        disabled={currentUserRole === 'admin' && option.id === 'create_super_admin'}
                        checked={permissions.includes(option.id)}
                        onChange={(e) => {
                          if (e.target.checked) setPermissions([...permissions, option.id]);
                          else setPermissions(permissions.filter(p => p !== option.id));
                        }}
                        className="w-4 h-4 rounded border-[var(--border)] text-[var(--primary)] focus:ring-[var(--primary)] bg-[var(--bg-elevated)]"
                      />
                      <span className="text-sm text-[var(--text-muted)] group-hover:text-[var(--text)] transition-colors">{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="pt-4 flex items-center gap-3">
                <button type="button" onClick={() => setIsRoleModalOpen(false)} className="flex-1 px-4 py-2 rounded-lg font-medium text-[var(--text)] border border-[var(--border)] hover:bg-[var(--bg-hover)] transition-colors">Cancel</button>
                <button type="submit" className="flex-1 px-4 py-2 rounded-lg font-medium text-white bg-[var(--primary)] hover:bg-[var(--primary-hover)] transition-colors">
                  {roleMode === 'create' ? 'Create Role' : 'Save Changes'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal Overlay for Create/Edit User */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-[fade-in_0.15s_ease-out]">
          <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl shadow-2xl w-full max-w-md overflow-hidden transform scale-100 transition-all">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] bg-[var(--bg)]">
              <h3 className="text-lg font-semibold text-[var(--text)]">
                {modalMode === 'create' ? 'Create New Account' : 'Edit Account'}
              </h3>
              <button
                onClick={() => setIsModalOpen(false)}
                className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)] rounded-md transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                  placeholder="Enter username"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 pr-10 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                    placeholder="Enter password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute inset-y-0 right-0 flex items-center pr-3 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                  placeholder="Enter email id"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Phone Number</label>
                <input
                  type="text"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                  pattern="[0-9]{10}"
                  maxLength={10}
                  className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                  placeholder="Enter 10-digit phone number"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Role</label>

                <select
                  value={role}
                  onChange={(e) => {
                    const newRole = e.target.value;
                    setRole(newRole as any);
                    const matchedRole = customRoles.find(r => r.name === newRole);
                    if (matchedRole) setPermissions(matchedRole.permissions);
                  }}
                  className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors appearance-none"
                >
                  {customRoles
                    .filter(r => currentUserRole === 'admin' ? (r.name !== 'super_admin' && r.name !== 'admin') : true)
                    .map(r => (
                      <option key={r.name} value={r.name}>{r.name.replace('_', ' ')}</option>
                    ))}
                </select>

              </div>

              <div className="pt-2">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-[var(--text-secondary)]">Sidebar Access</label>
                  <button
                    type="button"
                    onClick={() => {
                      const selectableOptions = currentUserRole === 'admin'
                        ? SIDEBAR_OPTIONS.filter(o => o.id !== 'create_super_admin')
                        : SIDEBAR_OPTIONS;
                      if (permissions.length >= selectableOptions.length) {
                        setPermissions([]);
                      } else {
                        setPermissions(selectableOptions.map(o => o.id));
                      }
                    }}
                    className="text-xs text-[var(--primary)] hover:underline"
                  >
                    {permissions.length === SIDEBAR_OPTIONS.length ? 'Deselect All' : 'Select All'}
                  </button>
                </div>
                <div className="bg-[var(--bg)] border border-[var(--border)] rounded-lg p-3 grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                  {SIDEBAR_OPTIONS.map(option => (
                    <label key={option.id} className="flex items-center gap-2 cursor-pointer group">
                      <input
                        type="checkbox"
                        disabled={currentUserRole === 'admin' && option.id === 'create_super_admin'}
                        checked={permissions.includes(option.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setPermissions([...permissions, option.id]);
                          } else {
                            setPermissions(permissions.filter(p => p !== option.id));
                          }
                        }}
                        className="w-4 h-4 rounded border-[var(--border)] text-[var(--primary)] focus:ring-[var(--primary)] bg-[var(--bg-elevated)]"
                      />
                      <span className="text-sm text-[var(--text-muted)] group-hover:text-[var(--text)] transition-colors">
                        {option.label}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="pt-4 flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="flex-1 px-4 py-2 rounded-lg font-medium text-[var(--text)] border border-[var(--border)] hover:bg-[var(--bg-hover)] transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 rounded-lg font-medium text-white bg-[var(--primary)] hover:bg-[var(--primary-hover)] transition-colors"
                >
                  {modalMode === 'create' ? 'Create' : 'Save Changes'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
